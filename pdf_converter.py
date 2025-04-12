import os
from pypdf import PdfReader

PATH = 'test_data/'

for file in os.listdir(PATH):
    reader = PdfReader(os.path.join(PATH, file))
    for page in reader.pages:
        with open(f'txt_data/{os.path.splitext(file)[0]}.txt', 'a') as new_file:
            new_file.write(page.extract_text() + '\n')
