import os
from dotenv import load_dotenv

import chromadb
from google import genai
from embedding_function import GeminiEmbeddingFunction


load_dotenv()

CHROMA_PATH = r'part_installation_db'

google_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
chroma_client = chromadb.PersistentClient(CHROMA_PATH)


def get_query_result(query_):
    fn = GeminiEmbeddingFunction()
    fn.document_mode = False
    r = chroma_client.get_collection(name='building_pcs', embedding_function=fn).query(query_texts=[query_],
                                                                                         n_results=5)
    [ans] = r['documents']
    return ans


# Search the Chroma DB using the specified query.
query = 'Please provide numbered instructions on how to install a hard drive'

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = False
result = chroma_client.get_collection(name='building_pcs', embedding_function=embed_fn).query(query_texts=[query],
                                                                                              n_results=10)
[all_passages] = result['documents']

query_oneline = query.replace('\n', ' ')

# This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
prompt = f'''You are a helpful and informative bot that answers questions related to PC hardware using the provided 
text. Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you are talking to a non-technical audience, so be sure to break down complicated concepts, strike a friendly 
and conversational tone, and mention that the users should always consult the related manuals. The user may not have the
manual on hand, suggest they may find a digital copy here: https://www.manuals.co.uk/computers-and-accessories. If the 
passage is irrelevant to the answer, you may ignore it. You may also use your own knowledge to support the text however,
it must be relevant. Make sure to mention any tools that are needed and remove any brand names that are referenced. If 
you are asked a question related to installing a piece of hardware or modifying the hardware in the PC be sure to remind 
the user to turn off and unplug the power cord. Should a question be asked that is not related to a PC or its hardware 
respond with: "I'm sorry but that is outside of my scope, please only ask questions related to PC's and their hardware. 
Is there anything else I can help you with?"

QUESTION: {query_oneline}
'''

# Add the retrieved documents to the prompt.
for passage in all_passages:
    passage_oneline = passage.replace('\n', ' ')
    prompt += f'PASSAGE: {passage_oneline}\n'

print(prompt)

answer = google_client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt)

print('\n\n', answer.text)
