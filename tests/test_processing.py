# We defined the features required at the top level of our test
import sys
import os
from pathlib import Path
import pandas as pd

import pytest

# Needed for pytest to resolve imports properly
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from ml_editor.data_ingestion import parse_xml_to_csv
from ml_editor.data_processing import (
    get_random_train_test_split,
    get_split_by_author,
    add_text_features_to_df,
    format_raw_df,
)

REQUIRED_FEATURES = [
    "is_question",
    "action_verb_full",
    "language_question",
    "question_mark_full",
    "text_len",
]
CURR_PATH = Path(os.path.dirname(__file__))
XML_PATH = Path("fixtures/MiniPosts.xml")
CSV_PATH = Path("fixtures/MiniPosts.csv")


# Make sure we have a csv
@pytest.fixture(scope="session", autouse=True)
def get_csv():
    parse_xml_to_csv(CURR_PATH / XML_PATH, save_path=CURR_PATH / CSV_PATH)


@pytest.fixture
def df_with_features():
    df = pd.read_csv(CURR_PATH / CSV_PATH)
    df = format_raw_df(df.copy())
    return add_text_features_to_df(df.copy())


def test_random_split_proportion():
    df = pd.read_csv(CURR_PATH / CSV_PATH)
    train, test = get_random_train_test_split(df, test_size=0.3)
    print(len(train), len(test))
    assert float(len(train) / 0.7) == float(len(test) / 0.3)


def test_author_split_no_leakage():
    df = pd.read_csv(CURR_PATH / CSV_PATH)
    train, test = get_split_by_author(df, test_size=0.3)
    train_owners = set(train["OwnerUserId"].values)
    test_owners = set(test["OwnerUserId"].values)
    assert len(train_owners.intersection(test_owners)) == 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_feature_presence(df_with_features):
    for feat in REQUIRED_FEATURES:
        assert feat in df_with_features.columns


def test_feature_type(df_with_features):
    assert df_with_features["is_question"].dtype == bool
    assert df_with_features["action_verb_full"].dtype == bool
    assert df_with_features["language_question"].dtype == bool
    assert df_with_features["question_mark_full"].dtype == bool
    assert df_with_features["text_len"].dtype == int


def test_text_length(df_with_features):
    text_mean = df_with_features["text_len"].mean()
    text_max = df_with_features["text_len"].max()
    text_min = df_with_features["text_len"].min()
    assert text_mean in pd.Interval(left=200, right=1000)
    assert text_max in pd.Interval(left=0, right=10000)
    assert text_min in pd.Interval(left=0, right=1000)
