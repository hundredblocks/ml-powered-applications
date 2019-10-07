import os
from pathlib import Path

from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ElT
import pandas as pd


def parse_xml_to_csv(path, save_path=None):
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

    # Create dataframe from our list of dictionaries
    df = pd.DataFrame.from_dict(all_rows)
    if save_path:
        df.to_csv(save_path)
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
