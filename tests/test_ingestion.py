import os

from data_transformation import parse_xml_to_csv, get_random_train_test_split
from pathlib import Path
import pandas as pd


def get_fixture_df():
    curr_path = os.path.dirname(__file__)
    return parse_xml_to_csv(curr_path / Path("test_data/MiniPosts.xml"))


def test_parser_returns_dataframe():
    df = get_fixture_df()
    assert isinstance(df, pd.DataFrame)


def test_text_present():
    raise NotImplementedError


def test_have_questions():
    raise NotImplementedError
