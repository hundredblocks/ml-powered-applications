import sys, os

from pathlib import Path
import pandas as pd

# Needed for pytest to resolve imports properly
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from data_ingestion import parse_xml_to_csv

TEXT_LENGTH_FIELD = "text_len"

# We defined the features required at the top level of our test
REQUIRED_COLUMNS = [
    "Id",
    "AnswerCount",
    "PostTypeId",
    "AcceptedAnswerId",
    "Body",
    "body_text",
    "Title",
    "Score",
    TEXT_LENGTH_FIELD,
]

# Acceptable interval created based on data exploration
ACCEPTABLE_TEXT_LENGTH_MEANS = pd.Interval(left=20, right=2000)


def get_fixture_df():
    """
    Use parser to return DataFrame
    :return:
    """
    curr_path = Path(os.path.dirname(__file__))
    return parse_xml_to_csv(curr_path / Path("fixtures/MiniPosts.xml"))


def test_parser_returns_dataframe():
    """
    Tests that our parser runs and returns a DataFrame
    """
    df = get_fixture_df()
    assert isinstance(df, pd.DataFrame)


def test_feature_columns_exist():
    """
    Validate that all required columns are present
    """
    df = get_fixture_df()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_features_not_all_null():
    """
    Validate that no features are missing every value
    """
    df = get_fixture_df()
    for col in REQUIRED_COLUMNS:
        assert not df[col].isnull().all()


def test_text_mean():
    """
    Validate that text mean matches with exploration expectations
    """
    df = get_fixture_df()
    text_col_mean = df[TEXT_LENGTH_FIELD].mean()
    assert text_col_mean in ACCEPTABLE_TEXT_LENGTH_MEANS
