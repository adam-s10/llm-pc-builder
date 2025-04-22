import os
import time

import streamlit as st
from google import genai
from embedding_function import GeminiEmbeddingFunction
import chromadb

# Setup webpage
st.title('PC Hardware Installation Assistant')
st.text('''
    Use our PC Hardware Installation Assistant to have all your questions related to the installation of your new 
    hardware answered. Powered by Google Gemini it can answer questions related to PC components but is specially 
    trained, using Retrieval-Augmented Generation, to answer specific questions about how to install individual components.
    '''
)


CHROMA_PATH = r'part_installation_db'
google_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
chroma_client = chromadb.PersistentClient(CHROMA_PATH)

def get_api_key():
    """Get API key from either env or sidebar input"""
    pass

def stream_text(text):
    for letter in text:
        time.sleep(0.005)
        yield letter

def get_query_result(query_, n_results):
    fn = GeminiEmbeddingFunction()
    fn.document_mode = False
    r = (chroma_client.get_collection(name='building_pcs', embedding_function=fn)
         .query(query_texts=[query_], n_results=n_results))
    [ans] = r['documents']
    return ans

def get_llm_response(query, passages):
    # This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
    prompt = f'''You are a helpful and informative bot that answers questions related to PC hardware using the provided 
        text. Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts (using 
        numbered steps where appropriate), strike a friendly and conversational tone, and mention that the users should 
        always consult the related manuals. The user may not have the manual on hand, suggest they may find a digital 
        copy here: https://www.manuals.co.uk/computers-and-accessories. If the passage is irrelevant to the answer, 
        you may ignore it. You may also use your own knowledge to support the text however, it must be relevant. Make 
        sure to mention any tools that are needed and remove any brand names that are referenced. If you are asked a 
        question related to installing a piece of hardware or modifying the hardware in the PC be sure to remind the 
        user to turn off and unplug the power cord. Should a question be asked that is not related to a PC or its 
        hardware respond with: "I'm sorry but that is outside of my scope, please only ask questions related to PC's 
        and their hardware. Is there anything else I can help you with?"

        QUESTION: {query}
        '''
    # Add the retrieved documents to the prompt.
    for passage in passages:
        passage_oneline = passage.replace('\n', ' ')
        prompt += f'PASSAGE: {passage_oneline}\n'

    return google_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
     )

if 'messages' not in st.session_state:
    st.session_state.messages = []
    welcome_message = 'Hello, how can I help you with your PC today?'
    st.session_state.messages.append({'role': 'assistant', 'content': welcome_message})

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if user_input := st.chat_input('What PC question do you have today?'):
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    query_oneline = user_input.replace('\n', ' ')

    all_passages = get_query_result(query_oneline, 10)
    response = get_llm_response(query_oneline, all_passages)

    with st.chat_message('assistant'):
        st.write_stream(stream_text(response.text))

    st.session_state.messages.append({'role': 'assistant', 'content': response.text})

# TODO: use command streamlit run app.py to run the program using localhost
