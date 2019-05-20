import os
from pathlib import Path

from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ElT
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from ml_editor import preprocess_input


def parse_xml_to_csv(path):
    """
    Open .xml posts dump and convert the text to a csv, tokenizing it in the process
    :param path: path to the xml document containing posts
    :return: a dataframe of processed text
    """

    # Use python's standard library to parse XML file
    doc = ElT.parse(path)
    root = doc.getroot()

    # Each row is a question
    all_rows = [row.attrib for row in root.findall("row")]

    # Using tdqm to display progress since preprocessing takes time
    for item in tqdm(all_rows):
        # Decode text from HTML
        soup = BeautifulSoup(item["Body"], features="html.parser")
        item["body_text"] = soup.get_text()

        # Tokenize text using our preprocessing function
        item["tokenized"] = preprocess_input(item["body_text"])
        num_words = len([word for sent in item["tokenized"] for word in sent])
        item["text_len"] = num_words
    # Create dataframe from our list of dictionaries
    df = pd.DataFrame.from_dict(all_rows)
    return df


def get_data_from_dump(site_name, load_existing=True):
    """
    load .xml dump, parse it to a csv, serialize it and return it
    :param load_existing: should we load the existing extract or regenerate it
    :param site_name: name of the stackexchange website
    :return: pandas DataFrame of the parsed xml
    """
    data_path = Path("data")
    dump_name = "%s.stackexchange.com/Posts.xml" % site_name
    extracted_name = "%s.csv" % site_name
    dump_path = data_path / dump_name
    extracted_path = data_path / extracted_name

    if not (load_existing and os.path.isfile(extracted_path)):
        all_data = parse_xml_to_csv(dump_path)
        all_data.to_csv(extracted_path)
    else:
        all_data = pd.DataFrame.from_csv(extracted_path)

    return all_data


def get_random_train_test_split(posts, test_size=0.3, random_state=40):
    """
    Get train/test split from DataFrame
    Assumes the DataFrame has one row per question example
    :param posts: all posts, with their labels
    :param test_size: the proportion to allocate to test
    :param random_state: a random seed
    """
    return train_test_split(
        posts, test_size=test_size, random_state=random_state
    )


def get_split_by_author(
    posts, author_id_column="OwnerUserId", test_size=0.3, random_state=40
):
    """
    Get train/test split
    Guarantee every author only appears in one of the splits
    :param posts: all posts, with their labels
    :param author_id_column: name of the column containing the author_id
    :param test_size: the proportion to allocate to test
    :param random_state: a random seed
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    splits = splitter.split(posts, groups=posts[author_id_column])
    return next(splits)


def get_split_by_author_and_time(
    posts,
    author_id_column="OwnerUserId",
    posted_column="CreationDate",
    test_size=0.3,
    random_state=40,
):
    """
    Get train/test split,
    Guarantee every author only appears in one of the splits
    Guarantee all training examples happen before test ones
    :param posts: all posts, with their labels
    :param posted_column: name of the column containing posted date
    :param author_id_column: name of the column containing the author_id
    :param test_size: the proportion to allocate to test
    :param random_state: a random seed
    """
    raise NotImplementedError()
