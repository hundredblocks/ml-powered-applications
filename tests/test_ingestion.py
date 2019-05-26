import sys, os

from pathlib import Path
import pandas as pd

# Needed for pytest to resolve imports properly
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from data_ingestion import parse_xml_to_csv

# We defined the features required at the top level of our test
REQUIRED_COLUMNS = [
    "AnswerCount",
    "PostTypeId",
    "AcceptedAnswerId",
    "Body",
    "text_len",
]

# Acceptable interval created based on data exploration
ACCEPTABLE_TEXT_LENGTH_MEANS = pd.Interval(left=20, right=2000)


def get_fixture_df():
    curr_path = Path(os.path.dirname(__file__))
    return parse_xml_to_csv(curr_path / Path("fixtures/MiniPosts.xml"))


def test_parser_returns_dataframe():
    df = get_fixture_df()
    assert isinstance(df, pd.DataFrame)


def test_feature_columns_exist():
    df = get_fixture_df()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_features_not_all_null():
    df = get_fixture_df()
    for col in REQUIRED_COLUMNS:
        assert not df[col].isnull().all()


def test_text_mean():
    df = get_fixture_df()
    text_col_mean = df["text_len"].mean()
    assert text_col_mean in ACCEPTABLE_TEXT_LENGTH_MEANS
