import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

search_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.0,
    )

chat = client.chats.create(
    model='gemini-2.0-flash',
    config=search_config
)

new_prompt = '''
        You are a computer hardware specialist who will help a user search for pre-built laptops or desktops based off 
        certain requirements that will be provided individually in the following messages. For each requirement you will 
        search online for hardware that fulfills the requirements, for example, minimum and recommended system specs. 
        You are to  use only information that is returned, do not make any assumptions. You should focus your search to 
        the UK and assume that the budget is in GBP (£). You will be told when all the requirements 
        have been exhausted when you receive the following message: "INTERNAL SYSTEM MESSAGE: there are no more 
        requirements". When you receive this message you should not acknowledge it and instead begin completing the 
        following tasks.

        Once you have been told there are no more requirements, using the information you have collected you will then 
        search for a pre-built desktop or laptop that fits the information you have collected to then suggest to the 
        user. You should suggest at least 3 options and return the results in JSON format with the name, price, 
        specifications, and a purchase link for the device. 

        JSON Structure to follow:
        Return a list of objects, where each object has three fields: name, price, and specifications.
        '''

# use first value as budget and remainder as requirements
imaginary_user_input = [600, 'Must support rainbow six siege at 1080p and 60fps',
                        'Must have at least 500GB of storage']

response = chat.send_message(new_prompt)
print(response.text)

while len(imaginary_user_input):
    if len(imaginary_user_input) == 1:
        val, imaginary_user_input = 'Budget is ' + str(imaginary_user_input[-1]), imaginary_user_input[:-1]
    else:
        val, imaginary_user_input = imaginary_user_input[-1], imaginary_user_input[:-1]
    other_responses = chat.send_message(
        message=val,
        config=search_config
    )
    print(other_responses.text)

final_response = chat.send_message('INTERNAL SYSTEM MESSAGE: there are no more requirements')

print(final_response.text)

# TODO: try using a chat that will be hidden to the user but could supply the provided information in chunks so
#  the google searches are performed one at a time eg: requirement searched one at a time

# while not rc.grounding_metadata.grounding_supports or not rc.grounding_metadata.grounding_chunks:
#     # If incomplete grounding data was returned, retry.
#     rc = query_with_grounding()
#
# chunks = rc.grounding_metadata.grounding_chunks
# for chunk in chunks:
#     print(f'{chunk.web.title}: {chunk.web.uri}')