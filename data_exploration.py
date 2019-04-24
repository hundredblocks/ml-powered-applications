import json

from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ElT
import matplotlib.pyplot as plt
import pandas as pd

from ml_editor import preprocess_input


def parse_doc_to_text(path):
    doc = ElT.parse(path)
    root = doc.getroot()

    all_rows = [row.attrib for row in root.findall('row')]

    for item in tqdm(all_rows):
        soup = BeautifulSoup(item["Body"], features="html.parser")
        item["Text"] = soup.get_text()
        item["Tokenized"] = preprocess_input(item["Text"])
    return all_rows


load_existing = True

if not load_existing:
    all_data = parse_doc_to_text('data/datascience.stackexchange.com/Posts.xml')
    with open('data/extracted.json', mode='w') as f:
        json.dump(all_data, f)
else:
    with open('data/extracted.json', mode='r') as f:
        all_data = json.load(f)

sentence_len = [len([word for sent in item["Tokenized"] for word in sent]) for
                item in all_data]

fig = plt.figure()
fig.suptitle('Distribution of question length for sentences')
plt.xlabel('Words per question')
plt.ylabel('Number of questions')
plt.hist(sentence_len, bins=1000, log=False)

sentence_len_truncated = [a for a in sentence_len if a < 2000]
print(len(sentence_len))
print(len(sentence_len_truncated))
fig = plt.figure()
fig.suptitle(
    'Distribution of question length for sentences shorter than 2000 words')
plt.xlabel('Words per question')
plt.ylabel('Number of questions')
plt.hist(sentence_len_truncated, bins=200, log=False)
# plt.show()


all_data_df = pd.DataFrame.from_dict(all_data)
all_data_df['PostTypeId'] = all_data_df['PostTypeId'].astype(int)
print(all_data_df[all_data_df['PostTypeId'] == 1].describe())
