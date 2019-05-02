import os

from data_transformation import parse_xml_to_csv
from pathlib import Path
import pandas as pd


def test_parser_returns_dataframe():
    curr_path = os.path.dirname(__file__)
    df = parse_xml_to_csv(curr_path / Path('test_data/MiniPosts.xml'))
    assert isinstance(df, pd.DataFrame)
