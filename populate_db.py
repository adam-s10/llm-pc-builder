import chromadb
from chromadb import EmbeddingFunction, Embeddings, Documents
from google.genai import types

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google import genai
from google.api_core import retry

from dotenv import load_dotenv
import os

# TODO: get all info added to the db
# TODO: investigate trigger words for RAG add more documents to the vector db - we have a ok solution
# TODO: create csvs for suggesting the pc
# TODO: find prebuilt websites
# TODO: streamlit
# TODO: create git - llm-pc-builder

load_dotenv()

DATA_PATH = 'txt_data/'
print(os.listdir(DATA_PATH))
CHROMA_PATH = r'chroma_db'

google_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
chroma_client = chromadb.Client()

# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input_: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = 'retrieval_document'
        else:
            embedding_task = 'retrieval_query'

        response = google_client.models.embed_content(
            model='models/text-embedding-004',
            contents=input_,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [em.values for em in response.embeddings]


def clean_extracted_text(text):
    new_text = text.replace('~', '')
    new_text = new_text.replace('©', '')
    new_text = new_text.replace('_', '')
    new_text = new_text.replace(';:;', '')
    new_text = new_text.replace('®', '')
    new_text = new_text.replace('#', '')
    new_text = new_text.replace('@', '')
    return new_text


embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True
db = chroma_client.get_or_create_collection(name='building_pcs', embedding_function=embed_fn)

for file in os.listdir(DATA_PATH):
    if file.startswith('.'):
        continue

    print('File being worked on:', file)
    file_path = os.path.join(DATA_PATH, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
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

    # Process chunks in smaller batches to manage resources and API limits
    batch_size = 50
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

# Confirm that the data was inserted by looking at the database.
print(db.count())
# Peek at the data.
print(db.peek(1))

embed_fn.document_mode = False

# Search the Chroma DB using the specified query.
query = 'Please tell me how to install an M.2 SSD in numbered instruction format'

result = db.query(query_texts=[query], n_results=10)
[all_passages] = result['documents']

query_oneline = query.replace('\n', ' ')

# This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
prompt = f'''You are a helpful and informative bot that answers questions related to PC hardware using the provided 
text. Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
strike a friendly and conversational tone. If the passage is irrelevant to the answer, you may ignore it. You may also 
use your own knowledge to support the text however, it must be relevant. Make sure to mention any tools that are needed. 
Should a question be asked that is not related to a PC or its hardware respond with: "I'm sorry but that is outside of 
my scope, please only ask questions related to PC's and their hardware. Is there anything else I can help you with?"

QUESTION: {query_oneline}
'''

# TODO: add a reference to the website that contains the manuals when manuals are referenced in your answer and remove
#  any brand names mentioned in your answer

# Add the retrieved documents to the prompt.
for passage in all_passages:
    passage_oneline = passage.replace('\n', ' ')
    prompt += f'PASSAGE: {passage_oneline}\n'

print(prompt)

answer = google_client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt)

print('\n\n', answer.text)
