import chromadb

from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google import genai
from embedding_function import GeminiEmbeddingFunction

from dotenv import load_dotenv
import os

# TODO: get all info added to the db - done
# Ones embedded: SSD, HDD, M.2, CPU, CPU cooler (air and aio), RAM (info about OC included), GPU, fans, MB, PSU

# TODO: investigate trigger words for RAG add more documents to the vector db - we have a ok solution

# TODO: create csvs for suggesting the pc - claude/deepseek/perplexity

# TODO: find prebuilt websites
# TODO: streamlit

load_dotenv()

DATA_PATH = 'txt_data/'  # will be empty as files moved to embedded_txt_data/ after embedding complete
print(os.listdir(DATA_PATH))
CHROMA_PATH = r'chroma_db'

google_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
chroma_client = chromadb.PersistentClient(CHROMA_PATH)

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True
db = chroma_client.get_or_create_collection(name='building_pcs', embedding_function=embed_fn)

def clean_extracted_text(text):
    new_text = text.replace('~', '')
    new_text = new_text.replace('©', '')
    new_text = new_text.replace('_', '')
    new_text = new_text.replace(';:;', '')
    new_text = new_text.replace('®', '')
    new_text = new_text.replace('#', '')
    new_text = new_text.replace('@', '')
    new_text = new_text.replace(' ', '')
    return new_text

for file in os.listdir(DATA_PATH):
    if file.startswith('.'):
        continue

    print('File being worked on:', file)
    file_path = os.path.join(DATA_PATH, file)
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