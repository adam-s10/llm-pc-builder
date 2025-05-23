# TODO: uncomment when uploading to notebook!!!
# Remove conflicting packages from the base environment.
# !pip uninstall -qqy kfp jupyterlab libpysal thinc spacy fastai ydata-profiling google-cloud-bigquery google-generativeai

# Install langgraph and the packages needed.
# !pip install -qU 'langgraph==0.3.21' 'langchain-google-genai==2.1.2' 'langgraph-prebuilt==0.1.7'

# Imports
import os
import json
import re
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.api_core import retry

import chromadb
from chromadb import EmbeddingFunction, Embeddings, Documents

from pprint import pprint
from typing import List, Dict, Any, Annotated, Literal, Optional
from collections.abc import Iterable


from typing_extensions import TypedDict
from IPython.display import Image

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolMessage

from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Setup API key for Gemini
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

CHROMA_PATH = r'part_selection_db' # TODO: change this to Kaggle path on upload!!!
DATA_PATH = 'txt_data/' # TODO: change this to Kaggle path on upload!!!
chroma_client = chromadb.PersistentClient(CHROMA_PATH)


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    # Define a helper to retry when per-minute quota is reached.
    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input_: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = 'retrieval_document'
        else:
            embedding_task = 'retrieval_query'

        response = genai.Client(api_key=os.getenv('GOOGLE_API_KEY')).models.embed_content(
            model='models/text-embedding-004',
            contents=input_,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [em.values for em in response.embeddings]

def clean_extracted_text(text: str) -> str:
    new_text = text.replace('~', '')
    new_text = new_text.replace('©', '')
    new_text = new_text.replace('_', '')
    new_text = new_text.replace(';:;', '')
    new_text = new_text.replace('®', '')
    new_text = new_text.replace('#', '')
    new_text = new_text.replace('@', '')
    new_text = new_text.replace(' ', '')
    return new_text

def get_chunks(file_path: str) -> List[Document]:
    loader = TextLoader(file_path)
    raw_documents = loader.load()
    print(raw_documents)

    # splitting the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(chunks)
    return chunks

def batch_embed_chunks(file_path: str, chunks: List[Document], batch_size: int=50) -> None:
    # Process chunks in smaller batches to manage resources and API limits
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_documents = [clean_extracted_text(chunk.page_content) for chunk in batch_chunks]

        try:
            batch_embeddings = embed_fn(batch_documents)
        except Exception as exc:
            print(f'Error generating embeddings for batch {i // batch_size} from {file_path}: {exc}')
            continue

        batch_metadata = [chunk.metadata for chunk in batch_chunks]
        batch_ids = [
            f'{os.path.splitext(file)[0]}_chunk_{i + j}'
            for j in range(len(batch_chunks))
        ]

        db.add(
            documents=batch_documents,
            metadatas=batch_metadata,
            ids=batch_ids
        )
        print(f'Added batch {i // batch_size + 1} from {file_path} to ChromaDB.  Total: {db.count()}')


embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True
db = chroma_client.get_or_create_collection(name='building_pcs', embedding_function=embed_fn)

for file in os.listdir(DATA_PATH):
    if not file.endswith('.txt'):
        continue

    print('File being worked on:', file)
    path = os.path.join(DATA_PATH, file)

    c = get_chunks(path)
    batch_embed_chunks(path, c)

embed_fn.document_mode = False

def get_query_result(query: str) -> list[str]:
    fn = GeminiEmbeddingFunction()
    fn.document_mode = False
    r = chroma_client.get_collection(name='building_pcs', embedding_function=fn).query(query_texts=[query],
                                                                                         n_results=5)
    [ans] = r['documents']
    return ans


class PCBuilderState(TypedDict):
    """State representing the PC builder conversation."""

    # The chat conversation. This preserves the conversation history
    # between nodes. The `add_messages` annotation indicates to LangGraph
    # that state is updated by appending returned messages, not replacing
    # them.
    messages: Annotated[list, add_messages]

    # User requirements
    # requirements: Dict[str, Any]
    requirements: list[str]

    # Budget information
    budget: str

    # Current recommendations for parts
    recommendations: list[Dict[str, Any]]

    # Flag for whether the build is complete
    build_complete: bool

    # Flag to indicate waiting for user input
    # waiting_for_user: bool


# The system instruction defines how the chatbot is expected to behave and includes
# rules for when to call different functions, as well as rules for the conversation, such
# as tone and what is permitted for discussion.
PC_BUILDER_SYSINT = (
    'system',  # 'system' indicates the message is a system instruction.
    '''
You are a PC Builder Assistant, an expert in computer hardware and building custom PCs.
Your goal is to help users find the perfect PC build based on their budget and requirements.

You should:
1. Ask about their budget and use case (gaming, office work, content creation, etc.)
2. Determine if they need a custom build or pre-built system.
3. For pre-built systems, ask if desktops or laptops or either are preferred.
4. For custom builds, if they have existing hardware, ask what parts they want to upgrade.

Once you have a general idea of what the user is looking for, you must formulate that
into a structured list of concise individual requirements and call the update_plans tool,
providing the budget and list of requirements. For custom build devices, if the user
mentions wanting the PC for a specific task that is hardware intensive, such as playing a
video game or video editing or training AI models or such, then first use the
search_task_requirements tool to find the hardware requirements for the specified task,
and only then update the user requirements using the update_plans tool, with the extra
hardware requirements added to it.

For pre-builds, even if there are potentially-intensive task requirements, you can call
the update_plans tool directly, adding the task requirements as part of the main requirements.

Once the requirements are all logged by the tool, summarize the requirements back to the
user in a couple of sentences using the latest requirements list and budget returned by
your last call of the tool.

If the user wants to change something in their plans, send a new list of requirements back
to update_plans. If they want to start from scratch, use the clear_plan tool to remove all
budget and requirements, then walk through the requirements gathering steps again.

If the user confirms their requirements are final and they want a pre-built device, use
search_prebuilt tool to find pre-built devices fitting the user criteria, which will be
given to you in a structured JSON format. If the JSON has formatting errors and you cannot
understand it, recall the search_prebuilt tool. If the JSON can be understood, call the
rank_prebuilds tool and provide it the exact same JSON (it will not work without JSON!) to
get a final list of recommendations and then render this neatly in markdown for the user
to browse.

If the user confirms their requirements are final and they want a custom build, use the
lookup_parts_needed tool to get a list of the parts required. Once you have the parts,
call the search_custom_parts tool to get a list of parts, which should be in a structured
JSON format. If the JSON has formatting errors and you cannot understand it, recall the
search_custom_parts tool. If the JSON can be understood, call the rank_parts tool and
provide it the exact same JSON (it will not work without JSON!) to get a final list of
parts for the user. Render this neatly in markdown for the user to browse.

The user may have additional questions about the parts or building process, which you must
expand upon if asked.

If any of the tools are unavailable, let the user know instead of trying to call the tools.

Stay focused on PC building. If users ask about unrelated topics, gently redirect them.
''',
)

# This is the message with which the system opens the conversation.
WELCOME_MSG = ('''
Welcome to the PC Builder Assistant! (Type `q` to quit). I'll help you find the perfect computer 
based on your needs and budget. Could you tell me your budget and what you'll be using this PC for? (Gaming, office 
work, content creation, etc.)
''')

# LLM model definitions
MAIN_MODEL = 'gemini-2.0-flash'
llm = ChatGoogleGenerativeAI(model=MAIN_MODEL)
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

# Removes the start and end of JSON codeblock formatting that the LLMs occasionally wrap JSON strings with.
def strip_json_wrapper(json_str: str) -> str:
    return json_str.replace('```json\n', '').replace('\n```', '')

# Utility function to help with parsing prices from strings.
def parse_price(price_str: str) -> float:
    """Extract numeric price value from the provided string."""
    if price_str == 'N/A':
        return None

    # Extract digits and decimal value
    price_match = re.search(r'[\d,.]+', price_str)
    if not price_match:
        return 0.0

    # Remove commas and convert to float
    price_digits = price_match.group(0).replace(',', '')
    if price_digits and not price_digits.isspace():
        return float(price_digits)

# Stateless tool - Search online for pre-built devices

@tool
def search_prebuilt(budget: str, requirements: list[str]) -> str:
    """Search for pre-built desktops or laptops fulfilling the user criteria.
    Take the budget and requirements, and get Gemini to format it into a terse, concise query.
    Then use search grounding to look for it and return the output as structured JSON with
    price, brand, desktop/laptop, name, link to view more or buy
    """
    print('CALLED TOOL: search_prebuilt')

    newline_char = '\n'
    search_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.0
    )

    prompt = f'''
You are a computer hardware specialist who always responds in valid JSON. Search online
for pre-built laptops or desktops based on the requirements provided below and return
the results in JSON format with name, price, specifications, and purchase link for each
device. Find at least 3-6 options for the device in question.

JSON Structure to follow:
Return a list of device objects, where each object has three fields: name, price, and specifications.

Key points to note:
- Please answer the following question using ONLY information found in the provided web search results. Cite your sources for each statement or paragraph.
- Rely exclusively on the real-time search results to answer. For each device in the JSON list you make, indicate which search result supports it.

Device Requirements:
    1. Budget: {budget}
    2. Requirements: {newline_char + (newline_char.join([((' ' * 8) + '- ' + req) for req in requirements]))}
'''

    response = None

    response = client.models.generate_content(
        model=MAIN_MODEL,
        contents=prompt,
        config=search_config
    )
    rc = response.candidates[0]
    return rc.content.parts[0].text

# Stateless tool - Search hardware requirements for a particular task
@tool
def search_task_requirements(special_task_requirement: str) -> list[str]:
    """
    Search hardware requirements for a particular task.
    Takes in a string describing the task and returns a list of hardware specs.
    """
    print('CALLED TOOL: search_task_requirements')
    print('Special use case:', special_task_requirement)

    sys_prompt = '''
You are a computer hardware specialist. The user has some hardware intensive tasks they want
to perform and they want to know the hardware specs their device will need for them to be able
to do what they want. For gaming, it might be helpful to consider minimum and recommended specs.
Search online for these specs needed and formulate that into a structured, comma-separated list
of concise individual requirements. Respond ONLY WITH a concise list of requirements separated by
commas, add no preface or conclusion.

Key points to note:
- Please answer the following question using ONLY information found in the provided web search results.
- Rely exclusively on the real-time search results to answer. Do not provide anything that is not found online.

Example answer 1:
Windows 10/11 64-bit, AMD Ryzen 7 CPU 5700X CPU, 16GB RAM, AMD Radeon RX 6700 XT, 170 GB SSD

Example answer 2:
Windows 11 64-bit, 8GB RAM, Intel Core i7-10700 CPU, NVIDIA GeForce RTX 2080 (8GB VRAM) GPU

Example answer 3:
AMD Ryzen 5 3600 @ 3.6 GHz or Intel Core i7-8700K @ 3.7 GHz or better, 8 GB RAM, AMD RX 570 (4 GB) or NVIDIA GeForce GTX 1060 (6 GB) or better, 85.5 GB storage
'''

    response = client.models.generate_content(
        model=MAIN_MODEL,
        contents=f'The use case that the hardware is needed for: {special_task_requirement}',
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            system_instruction=sys_prompt,
            temperature=0.0
        )
    )

    rc = response.candidates[0]
    hardware_needed = rc.content.parts[0].text
    if len(hardware_needed) > 0:
        return hardware_needed.split(', ')
    return hardware_needed

# Stateless tool - Lookup parts required internally using RAG and rules of thumb
@tool
def lookup_parts_needed(requirements: Iterable[str]) -> list[str]:
    """
    Look to improve the suggested hardware from search_task_requirements using the rules outlined in system_prompt
    :return list of components that will meet the requirements and conform to the rules
    """
    print('CALLED TOOL: lookup_parts_needed')

    # TODO: add everything to this file that's used ie query_db.py etc
    # TODO: message kaggle forum about uploading embedding data and or txt files - @Gen AI Moderator - done
    # TODO: look into pastebin if Kaggle no worky - done
    # TODO: look into adding the embeddings for component installation tomorrow from human node

    static_prompt = '''
    You are a computer hardware specialist, able to provide PC hardware recommendations based off of 
    requirements you will be presented and using the provided text. When recommending parts you should conform to the 
    following rules:
    RULES:
    1. Select Graphics Processing Unit (GPU) that’s ~50% of the total budget
    2. Select a Central Processing Unit (CPU) that does not bottleneck GPU
    3. Select motherboard (MB) based on quality of above 2 components (budget vs pro build)
    4. Select Random Access Memory (RAM) related to Overclock (OC) specs on CPU and MB
         i. Don’t suggest RAM that has OC when MB/CPU does not support it, prioritize price in this situation
        ii. Ensure RAM is on MB Qualified Vendors List (QVL)
    5. Select a Power Supply Unit (PSU) that can support all components power draw in Watts (W)
         i. Prioritize modular power supplies should budget allow
        ii. Minimum 10% headroom over expected power draw to account for cooler/storage/fans etc
       iii. Include optional 20-30% headroom for future upgrades if budget allows
            a. Eg components expected power draw is 800W so suggest ~900W PSU
    6. Selecting Storage
         i. Prioritize traditional 2.5 Solid State Drives (SSDs) for budget builds with as much storage as possible
        ii. Suggest M.2 SSDs for pro builds
       iii. Hard Disk Drives (HDDs) only if minimum storage cannot be met within budget
    7. Select a cooler that is both appropriate for the build and budget 
         i. eg not air cooler for hotter CPUs (Intel 14900K, etc.)
    8. Select a case that will meet the GPU length requirements, MB type (ATX, ITX etc), and number of SSDs/HDDs suggested
    9. Select fans depending on number of fans that come with the case
         i. Aim to have minimum 3 (2 intake; 1 exhaust)
        ii. Should budget allow, fill all fan slots on case
    10. Any money left over from the budget should be used to improve the computer. For example, picking a better GPU, 
        CPU, more storage etc.
    
    You must respond with a list of components, separated by commas with no additional information or comment. This list
    must include a CPU, GPU, PSU, case, motherboard, fans, and storage unless the user has provided one of these already
    as existing hardware.
    '''

    dynamic_prompt = '''
    The parts recommended must be able to accommodate the following requirements and be supported by their passages:
    '''

    # Maybe try to get it in format Requirement: \n Passage: (might use too many tokens)
    for req in requirements:
        static_prompt += f'REQUIREMENT: {req}\n'
        passages = []
        passages += get_query_result(req)

        for passage in passages:
            passage_oneline = passage.replace('\n', ' ')
            static_prompt += f'PASSAGE: {passage_oneline}\n'
        print(static_prompt)

    answer = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=dynamic_prompt,
        config=types.GenerateContentConfig(
            system_instruction=static_prompt,
            temperature=0.2
        )
    )
    # print(answer)
    answer = answer.text
    answer = answer.replace('\n', '')
    print(answer)
    return [s for s in answer.split(', ')]

# All stateless tools, part of tools node: Search game specs, lookup parts needed (sends to
# chatbot which then sends to search part then back to chatbot which then updates
# recommendations; must also take in existing hardware), search custom part (searches
# each part mentioned by lookup parts, takes in list of strs from chatbot or directly
# from lookup, does google search or external api call,  returns json for each part) ->
# this then goes to optimize build and then gets put in recommendations.

# Utility function
def search_individual_part(budget: str, part_needed: str) -> tuple[str, Any]:
    """
    """
    print(f'CALLED TOOL-HELPER: search_individual_part: {part_needed}')

    newline_char = '\n'

    sys_prompt = '''
You are a computer hardware specialist who always responds in valid JSON. Search online
for computer hardware parts that match the part that the user is looking for. Be sure to
bear the budget constraints in mind when looking. Recommend the latest parts that match
this criteria and return the results in JSON format with name, price, specifications,
and purchase link for each part. Find at least 2-4 options for this piece of hardware.

JSON Structure to follow:
Return a list of part objects, where each object has four fields: name, price, specifications,
and purchase link.

Key points to note:
- You should focus your search to the US and assume that the budget is in USD ($).
- Please answer the following question using ONLY information found in the provided web search results.
- Rely exclusively on the real-time search results to answer.
- Respond only in concise JSON as requested, do not add any preface or conclusions.
'''

    search_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.0,
        system_instruction=sys_prompt
    )

    response = client.models.generate_content(
        model=MAIN_MODEL,
        contents=f'Look for {part_needed} for a budget of {budget}.',
        config=search_config
    )
    rc = response.candidates[0]
    part_recommendations = rc.content.parts[0].text
    part_recommendations = strip_json_wrapper(part_recommendations)
    print(part_recommendations)

    return (part_needed, json.loads(part_recommendations))

# Parts scorer function
def extract_specs_score(specs: str, part_type: str) -> float:
    """
    Extract a numerical score from specifications based on part type.
    Higher score means better performance.
    """
    score = 0
    specs_lower = specs.lower()

    # CPU scoring
    if any(cpu_term in part_type.lower() for cpu_term in ['cpu', 'processor', 'core']):
        # Score based on cores
        core_match = re.search(r'(\d+)\s*cores?', specs, re.IGNORECASE) or re.search(r'(\d+)[\s-]*core', specs, re.IGNORECASE)
        if core_match:
            score += int(core_match.group(1)) * 10

        # Score based on threads
        thread_match = re.search(r'(\d+)\s*threads?', specs, re.IGNORECASE)
        if thread_match:
            score += int(thread_match.group(1)) * 5

        # Score based on clock speed
        clock_match = re.search(r'(\d+\.\d+)\s*GHz', specs, re.IGNORECASE)
        if clock_match:
            score += float(clock_match.group(1)) * 20

        # Score based on cache
        cache_match = re.search(r'(\d+)\s*MB\s*[Cc]ache', specs)
        if cache_match:
            score += int(cache_match.group(1)) * 2

        # Bonus for newer generations
        if '14' in specs:
            score += 50
        elif '13' in specs:
            score += 40
        elif '12' in specs:
            score += 30
        elif '11' in specs:
            score += 20
        elif '10' in specs:
            score += 10

    # GPU scoring
    elif any(gpu_term in part_type.lower() for gpu_term in ['gpu', 'graphics', 'video card', 'geforce', 'radeon']):
        # Score based on CUDA cores or Stream Processors
        cuda_match = re.search(r'(\d+,?\d*)\s*CUDA', specs, re.IGNORECASE)
        if cuda_match:
            cuda_cores = int(cuda_match.group(1).replace(',', ''))
            score += cuda_cores / 100

        stream_match = re.search(r'(\d+,?\d*)\s*Stream\s*Processors', specs, re.IGNORECASE)
        if stream_match:
            stream_processors = int(stream_match.group(1).replace(',', ''))
            score += stream_processors / 80

        # Score based on memory
        memory_match = re.search(r'(\d+)\s*GB', specs, re.IGNORECASE)
        if memory_match:
            score += int(memory_match.group(1)) * 10

        # Score based on memory speed
        speed_match = re.search(r'(\d+\.\d+)\s*Gbps', specs, re.IGNORECASE)
        if speed_match:
            score += float(speed_match.group(1)) * 2

        # Score based on memory bus width
        bus_match = re.search(r'(\d+)[\s-]*bit', specs, re.IGNORECASE)
        if bus_match:
            score += int(bus_match.group(1)) / 10

        # Score based on clock speeds
        clock_match = re.search(r'(\d+)\s*MHz', specs, re.IGNORECASE)
        if clock_match:
            score += int(clock_match.group(1)) / 100

    # RAM scoring
    elif any(ram_term in part_type.lower() for ram_term in ['ram', 'memory', 'ddr']):
        # Score based on capacity
        capacity_match = re.search(r'(\d+)\s*GB', specs, re.IGNORECASE)
        if capacity_match:
            score += int(capacity_match.group(1)) * 10

        # Score based on speed
        speed_match = re.search(r'(\d+)\s*MHz', specs, re.IGNORECASE) or re.search(r'DDR\d+-(\d+)', specs, re.IGNORECASE)
        if speed_match:
            score += int(speed_match.group(1)) / 100

        # Score based on CAS latency
        cas_match = re.search(r'CL(\d+)', specs, re.IGNORECASE)
        if cas_match:
            # Lower CAS is better, so we invert the relationship
            score += 20 - int(cas_match.group(1))

        # Type of RAM bonus
        if 'ddr5' in specs_lower:
            score += 50
        elif 'ddr4' in specs_lower:
            score += 30
        elif 'ddr3' in specs_lower:
            score += 10

    # Storage scoring
    elif any(storage_term in part_type.lower() for storage_term in ['ssd', 'hdd', 'storage', 'drive', 'nvme']):
        # Score based on capacity
        tb_match = re.search(r'(\d+)\s*TB', specs, re.IGNORECASE)
        if tb_match:
            score += int(tb_match.group(1)) * 100

        gb_match = re.search(r'(\d+)\s*GB', specs, re.IGNORECASE)
        if gb_match:
            score += int(gb_match.group(1)) / 10

        # Score based on read/write speeds
        read_match = re.search(r'(\d+,?\d*)\s*MB\/s read', specs, re.IGNORECASE) or re.search(r'read:?\s*(\d+,?\d*)\s*MB\/s', specs, re.IGNORECASE)
        if read_match:
            score += int(read_match.group(1).replace(',', '')) / 100

        # Type bonus
        if any(fast_storage in specs_lower for fast_storage in ['nvme', 'm.2', 'pcie 4.0', 'pcie4']):
            score += 50
        elif 'ssd' in specs_lower:
            score += 30
        elif 'hdd' in specs_lower:
            score += 10

    # Motherboard scoring
    elif any(mb_term in part_type.lower() for mb_term in ['motherboard', 'mainboard', 'mobo']):
        # Score based on chipset/generation
        chipset_score = 0
        if any(x in specs_lower for x in ['z790', 'x670', 'x670e']):
            chipset_score = 50
        elif any(x in specs_lower for x in ['z690', 'x570', 'b650']):
            chipset_score = 40
        elif any(x in specs_lower for x in ['b550', 'z590', 'b560']):
            chipset_score = 30
        score += chipset_score

        # Score based on memory support
        if 'ddr5' in specs_lower:
            score += 30
        elif 'ddr4' in specs_lower:
            score += 15

        # Score based on PCIe support
        if 'pcie 5.0' in specs_lower or 'pcie5' in specs_lower:
            score += 30
        elif 'pcie 4.0' in specs_lower or 'pcie4' in specs_lower:
            score += 20
        elif 'pcie 3.0' in specs_lower or 'pcie3' in specs_lower:
            score += 10

        # Score based on connectivity
        if 'wifi' in specs_lower:
            score += 15
        if 'bluetooth' in specs_lower:
            score += 10
        if 'usb 3' in specs_lower or 'usb-c' in specs_lower:
            score += 15

    # PSU scoring
    elif any(psu_term in part_type.lower() for psu_term in ['psu', 'power supply', 'power']):
        # Score based on wattage
        wattage_match = re.search(r'(\d+)\s*W', specs, re.IGNORECASE) or re.search(r'(\d+)\s*watt', specs, re.IGNORECASE)
        if wattage_match:
            score += int(wattage_match.group(1)) / 10

        # Score based on efficiency
        if '80+ titanium' in specs_lower:
            score += 50
        elif '80+ platinum' in specs_lower:
            score += 40
        elif '80+ gold' in specs_lower:
            score += 30
        elif '80+ silver' in specs_lower:
            score += 20
        elif '80+ bronze' in specs_lower:
            score += 10

        # Score based on modularity
        if 'full modular' in specs_lower or 'fully modular' in specs_lower:
            score += 20
        elif 'semi modular' in specs_lower or 'semi-modular' in specs_lower:
            score += 10

    # Case scoring
    elif any(case_term in part_type.lower() for case_term in ['case', 'chassis', 'tower']):
        # Score based on form factor
        if 'full tower' in specs_lower:
            score += 30
        elif 'mid tower' in specs_lower:
            score += 20
        elif 'mini' in specs_lower:
            score += 10

        # Score based on features
        if 'tempered glass' in specs_lower:
            score += 15
        if 'rgb' in specs_lower:
            score += 10
        if 'usb-c' in specs_lower or 'usb 3' in specs_lower:
            score += 15

        # Score based on cooling support
        fan_match = re.search(r'(\d+)\s*fans?', specs, re.IGNORECASE)
        if fan_match:
            score += int(fan_match.group(1)) * 5

    # Cooling scoring
    elif any(cooling_term in part_type.lower() for cooling_term in ['fan', 'cooler', 'cooling', 'aio']):
        # Score based on size
        mm_match = re.search(r'(\d+)\s*mm', specs, re.IGNORECASE)
        if mm_match:
            score += int(mm_match.group(1)) / 10

        # Score based on type
        if 'aio' in specs_lower or 'liquid' in specs_lower or 'water' in specs_lower:
            score += 30
        elif 'air' in specs_lower:
            score += 15

        # Score based on RGB
        if 'rgb' in specs_lower:
            score += 10

    # Generic scoring for all part types based on specs length
    # If specific metrics aren't found, we'll assume more text means more features
    score += len(specs) * 0.01

    return score

def rank_options_available(parts_data: Dict[str, List[Dict[str, str]]], budget: float) -> List[Dict[str, Any]]:
    """
    Rank parts based on specifications and price, within budget.
    Returns a list of top three ranked parts with calculated metrics.
    """
    all_parts = []

    for part_type, parts in parts_data.items():
        for part in parts:
            price = parse_price(part['price'])
            if price is None:
                continue  # Skip parts with no price

            if price > budget:
                continue  # Skip parts over budget

            specs_score = extract_specs_score(part['specifications'], part_type)

            # Calculate value score (performance per dollar)
            value_score = specs_score / price if price > 0 else 0

            all_parts.append({
                'name': part['name'],
                'price': part['price'],
                'raw_price': price,
                'specifications': part['specifications'],
                'purchase link': part['purchase link'],
                'part_type': part_type,
                'specs_score': specs_score,
                'value_score': value_score
            })

    # Sort by specs score first (higher is better)
    all_parts.sort(key=lambda x: x['specs_score'], reverse=True)

    # Get top three parts
    top_parts = all_parts[:3]

    # Clean up the output by removing temporary fields
    for part in top_parts:
        part.pop('raw_price', None)
        part.pop('specs_score', None)
        part.pop('value_score', None)

    return top_parts


def process_parts(json_data: str, budget: float) -> str:
    """
    Main function to process parts data and return top three ranked parts.
    """
    try:
        # Parse JSON string to dictionary
        parts_dict = json.loads(json_data)

        # Rank parts
        top_parts = {}
        for part_type, part_data in parts_dict.items():
            top_parts[part_type] = rank_options_available({part_type: part_data}, budget)

        # Convert results back to JSON
        result_json = json.dumps(top_parts, indent=4, ensure_ascii=False)
        return result_json

    except json.JSONDecodeError:
        return json.dumps({'error': 'Invalid JSON input'})
    except Exception as e:
        return json.dumps({'error': str(e)})


# Stateless tool - Search online for custom parts for the build
@tool
def search_custom_parts(budget: str, parts_needed: Iterable[str] = []) -> Dict[str, list[Dict[str, Any]]]:
    """
    """
    print('CALLED TOOL: search_custom_parts')

    parts_found = {}
    for part_needed in parts_needed:
        part_type, part_data = search_individual_part(budget, part_needed)
        parts_found[part_type] = part_data

    return json.dumps(parts_found, indent=4, ensure_ascii=False)

# Tool signatures for planning the build
# Functionality defined in pc_planner_node
@tool
def update_plans(requirements: Iterable[str], budget: Optional[str] = None) -> Dict[str, Any]:
    """
    Adds or modifies the device requirements and budget.
    Returns a confirmation of the budget and requirements that were just added to state.
    """


@tool
def clear_plan():
    """
    Removes all requirements and budget information and resets to blank slate.
    """

# Tool signatures for recommending parts for a planned build
# Functionality defined in optimize_build_node
@tool
def rank_parts(recommended_parts: str = '') -> str:
    """
    Takes in a list of parts in JSON and a budget and returns the most
    performant yet price-optimal parts for each part.
    Modifies state by adding the parts to the recommendations list.
    """


@tool
def rank_prebuilds(recommended_devices: str = '') -> str:
    """
    Takes in a list of prebuilt devices in JSON and a budget and returns the top
    three most performant yet price-optimal devices, also in JSON.
    Modifies state by adding the devices to the recommendations list.
    """

# Helper functions required to rank pre-built devices

def score_specs(specs: str) -> int:
    """
    Analyze specifications to score device performance.
    Higher scores indicate better performance.
    """
    score = 0
    specs = specs.lower()

    # CPU scoring
    if 'ryzen 7' in specs or 'i7' in specs:
        score += 80
    elif 'ryzen 5' in specs or 'i5' in specs:
        score += 60
    elif 'ryzen 3' in specs or 'i3' in specs:
        score += 40

    # Generation bonus
    if '5700' in specs:
        score += 20
    elif '5600' in specs or '5500' in specs:
        score += 15
    elif '4500' in specs or '4600' in specs:
        score += 10

    # RAM scoring
    ram_match = re.search(r'(\d+)gb', specs.replace(' ', ''))
    if ram_match:
        ram_size = int(ram_match.group(1))
        if ram_size >= 32:
            score += 50
        elif ram_size >= 16:
            score += 30
        elif ram_size >= 8:
            score += 15

    # Storage scoring
    if '1tb' in specs.replace(' ', ''):
        score += 30
    elif '500gb' in specs.replace(' ', '') or '512gb' in specs.replace(' ', ''):
        score += 20

    if 'nvme' in specs or 'ssd' in specs:
        score += 20

    # GPU scoring
    if 'rtx 3080' in specs:
        score += 100
    elif 'rtx 3070' in specs:
        score += 80
    elif 'rtx 3060' in specs:
        score += 70
    elif 'gtx 1650' in specs:
        score += 40
    elif 'radeon' in specs or 'onboard' in specs:
        score += 20

    return score


def get_value_ratio(device: Dict[str, Any], budget: float) -> float:
    """
    Calculate value ratio based on specs score and price.
    Returns 0 if device exceeds budget.
    """
    price = parse_price(device['price'])
    if price > budget:
        return 0

    specs_score = score_specs(device['specifications'])

    # Calculate value ratio (specs score per unit of price)
    # Higher ratio means better value
    if price > 0:
        return specs_score / price
    return 0


def rank_devices(devices_json: str, budget: float, return_json: Optional[bool] = False) -> Any:
    """
    Rank devices based on specifications and price within budget.
    Returns JSON string with top 3 devices.
    """
    devices = json.loads(devices_json)
    if isinstance(devices, dict):
        devices = devices['devices']

    for device in devices:
        device['value_ratio'] = get_value_ratio(device, budget)

    # Sort devices by assigned value ratios descending
    ranked_devices = sorted(devices, key=lambda x: x['value_ratio'], reverse=True)

    # Select top 3 within budget
    top_devices = [
        {k: v for k, v in device.items() if k != 'value_ratio'}
        for device in ranked_devices[:3] if parse_price(device['price']) <= budget
    ]

    if return_json:
        return json.dumps(top_devices, indent=4, ensure_ascii=False)
    return top_devices

# Tool grouped on nodes
planner_tools = [update_plans, clear_plan]
builder_tools = [rank_parts, rank_prebuilds]

# Tools Config
auto_tools = [search_prebuilt, search_task_requirements, lookup_parts_needed, search_custom_parts]
tool_node = ToolNode(auto_tools)

# Tool binding
llm_with_tools = llm.bind_tools(auto_tools + planner_tools + builder_tools)

# Build planner node
def pc_planner_node(state: PCBuilderState) -> PCBuilderState:
    """This is where the requirements and budget within state get manipulated."""

    tool_msg = state.get("messages", [])[-1]
    requirements_state = state.get("requirements", [])
    budget_state = state.get('budget', None)
    recommendations_state = state.get('recommendations', [])
    outbound_msgs = []
    build_complete = state.get('build_complete', False)

    for tool_call in tool_msg.tool_calls:
        if tool_call['name'] == 'update_plans':
            print('CALLED TOOL: update_plans')
            args_given = tool_call['args']
            requirements_arg = budget_arg = None
            if 'requirements' in args_given:
                requirements_arg = args_given['requirements']
            if 'budget' in args_given:
                budget_arg = args_given['budget']
            # If budget is None and nothing exists in state, raise an error.
            # If there is something in state, then just don't update it.
            # Otherwise, always update the budget.
            if budget_arg is None or len(budget_arg) < 1:
                if budget_state is None or len(budget_state) < 1:
                    raise ValueError('Budget is missing in tool call as well as state!')
            else:
                budget_state = budget_arg
            if requirements_arg is None or len(requirements_arg) < 1:
                if requirements_state is None or len(requirements_state) < 1:
                    raise ValueError('Requirements are missing in tool call as well as state!')
            else:
                requirements_state = [requirement for requirement in requirements_arg]
            response = {
                'budget': budget_arg if budget_arg is not None else budget_state,
                'requirements': requirements_arg
            }

        elif tool_call['name'] == 'clear_plan':
            print('CALLED TOOL: clear_plan')
            requirements_state = []
            budget_state = None
            response = None

        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

        # Record the tool results as tool messages.
        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {
        'messages': outbound_msgs,
        'requirements': requirements_state,
        'budget': budget_state,
        'recommendations': recommendations_state,
        'build_complete': build_complete
    }

# Parts recommender node (based on devised plan and recommended devices)
def optimize_build_node(state: PCBuilderState) -> PCBuilderState:
    """This is where the recommendations within state get manipulated."""
    # state.recommendations modified by tool rank_parts() part of optimize_build node
    # - happens once search_prebuilt() tool returns a few values

    tool_msg = state.get('messages', [])[-1]
    requirements_state = state.get('requirements', [])
    budget_state = state.get('budget', None)
    recommendations_state = state.get('recommendations', [])
    outbound_msgs = []
    build_complete = state.get('build_complete', False)

    for tool_call in tool_msg.tool_calls:
        args_available = tool_call['args']
        if tool_call['name'] == 'rank_parts':
            print('CALLED TOOL: rank_parts')
            if 'recommended_parts' not in args_available:
                raise ValueError('There are no recommended parts available to rank!')
            recommended_parts = args_available['recommended_parts']
            if budget_state is None or len(budget_state) < 1:
                raise ValueError('An invalid budget was found within state!')
            print('\nThis is the custom parts recommendations json provided:\n', recommended_parts, '\n')
            recommendations_state = process_parts(strip_json_wrapper(recommended_parts), parse_price(budget_state))
            response = recommendations_state

        elif tool_call['name'] == 'rank_prebuilds':
            print('CALLED TOOL: rank_prebuilds')
            recommended_devices = tool_call['args']['recommended_devices']
            if budget_state is None or len(budget_state) < 1:
                raise ValueError('An invalid budget was found within state!')
            print('\nThis is the prebuilds recommendations json provided:\n', recommended_devices, '\n')
            recommendations_state = rank_devices(recommended_devices, parse_price(budget_state))
            response = rank_devices(recommended_devices, parse_price(budget_state), True)

        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

        # Record the tool results as tool messages.
        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_call['name'],
                tool_call_id=tool_call['id'],
            )
        )

    return {
        'messages': outbound_msgs,
        'requirements': requirements_state,
        'budget': budget_state,
        'recommendations': recommendations_state,
        'build_complete': build_complete
    }

def chatbot_node(state: PCBuilderState) -> PCBuilderState:
    """The chatbot itself. A simple wrapper around the model's own chat interface."""
    default_state = {'requirements': [], 'budget': None, 'recommendations': [], 'build_complete': False}

    if state['messages']:
        # If there are messages, continue the conversation with the model
        message_history = [PC_BUILDER_SYSINT] + state['messages']
        new_output = llm_with_tools.invoke(message_history)
    else:
        # If there are no messages, welcome the user.
        new_output = AIMessage(content=WELCOME_MSG)

    # Setup some defaults, then override with whatever exists in state, and finally
    # override with messages
    return default_state | state | {'messages': [new_output]}

def human_node(state: PCBuilderState) -> PCBuilderState:
    """Display the last message from the model to the user, and receive their input."""

    print('\nAll state until now:')
    for key, val in state.items():
        print(key, '::', val)
        print('\n')
    print('\nNow onto the messages:\n')

    last_msg = state['messages'][-1]
    print('Model:', last_msg.content)

    user_input = input('User: ')

    # Does the user wish to quit?
    if user_input in {'q', 'quit', 'exit', 'goodbye'}:
        state['build_complete'] = True

    return state | {'messages': [('user', user_input)]}

# Human to Exit OR Human to Chatbot; Conditional Edge Transition function
def maybe_exit_human_node(state: PCBuilderState) -> Literal["chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    if state.get('build_complete', False):
        return END
    else:
        return 'chatbot'

# Chatbot to Tools OR Chatbot to Human; Conditional Edge Transition function
def maybe_route_to_tools(state: PCBuilderState) -> str:
    if not (msgs := state.get('messages', [])):
        raise ValueError(f'No messages found when parsing state: {state}')

    # Only route based on the last message.
    msg = msgs[-1]

    if state.get('build_complete', False):
        # If the user has no more questions or indicates satisfaction, complete the build
        return END
    elif hasattr(msg, 'tool_calls') and len(msg.tool_calls) > 0:
        # When chatbot returns tool_calls, route to the 'tools' node
        if any(tool['name'] in tool_node.tools_by_name.keys() for tool in msg.tool_calls):
            return 'tools'
        elif any(tool['name'] in [func.name for func in planner_tools] for tool in msg.tool_calls):
            return 'pc_planner'
        elif any(tool['name'] in [func.name for func in builder_tools] for tool in msg.tool_calls):
            return 'optimize_build'
        else:
            raise ValueError('A nonexistent node was called.')
    else:
        return 'human'

# Set up the initial graph based on our state definition.
graph_builder = StateGraph(PCBuilderState)

# Add all the nodes to the app graph.
graph_builder.add_node('chatbot', chatbot_node)
graph_builder.add_node('human', human_node)
graph_builder.add_node('tools', tool_node)
graph_builder.add_node('pc_planner', pc_planner_node)
graph_builder.add_node('optimize_build', optimize_build_node)

# Define the chatbot node as the app entrypoint.
graph_builder.add_edge(START, 'chatbot')

# Edge transitions
graph_builder.add_conditional_edges('chatbot', maybe_route_to_tools)
graph_builder.add_conditional_edges('human', maybe_exit_human_node)
graph_builder.add_edge('tools', 'chatbot')
graph_builder.add_edge('pc_planner', 'chatbot')
graph_builder.add_edge('optimize_build', 'chatbot')

chat_graph = graph_builder.compile()

# Visualize the graph created.
# Image(chat_graph.get_graph().draw_mermaid_png())  # TODO: uncomment when uploading to notebook!!!

# The default recursion limit for traversing nodes is 25 - setting it higher means
# you can try a more complex order with multiple steps and round-trips (and you
# can chat for longer!)
config = {'recursion_limit': 100}

# Remember that this will loop forever, unless you input `q`, `quit` or one of the
# other exit terms defined in `human_node`.
# Uncomment this line to execute the graph:
state = chat_graph.invoke({'messages': []}, config)

# Things to try:
#  - Just chat! There's no ordering or menu yet.
#  - 'q' to exit.

pprint(state)
