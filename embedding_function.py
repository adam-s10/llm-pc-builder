from chromadb import EmbeddingFunction, Embeddings, Documents

from google import genai
from google.api_core import retry
from google.genai import types

import os
from dotenv import load_dotenv

load_dotenv()


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True
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