import json

from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ElT
import pandas as pd

from ml_editor import preprocess_input


def parse_posts_to_dataframe(path):
    """

    :param path: path to the xml document containing posts
    :return: a dataframe of processed text
    """

    # Use python's standard library to parse XML file
    doc = ElT.parse(path)
    root = doc.getroot()

    # Each row is a question
    all_rows = [row.attrib for row in root.findall('row')]

    # Using tdqm to display progress since preprocessing takes time
    for item in tqdm(all_rows):
        # Decode text from HTML
        soup = BeautifulSoup(item["Body"], features="html.parser")
        item["Text"] = soup.get_text()

        # Tokenize text using our preprocessing function
        item["Tokenized"] = preprocess_input(item["Text"])
        num_words = len([word for sent in item["Tokenized"] for word in sent])
        item["question_len"] = num_words

    # Create dataframe from our list of dictionaries
    df = pd.DataFrame.from_dict(all_rows)
    return df


load_existing = True
if not load_existing:
    all_data = parse_posts_to_dataframe(
        'data/datascience.stackexchange.com/Posts.xml')
    all_data.to_csv('data/extracted.csv')
else:
    all_data = pd.DataFrame.from_csv('data/extracted.csv')

all_data['PostTypeId'] = all_data['PostTypeId'].astype(int)
print(all_data[all_data['PostTypeId'] == 1].describe())
