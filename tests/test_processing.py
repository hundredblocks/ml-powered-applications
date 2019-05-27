# We defined the features required at the top level of our test
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

import pytest

# Needed for pytest to resolve imports properly
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from data_ingestion import parse_xml_to_csv
from data_processing import (
    get_random_train_test_split,
    get_split_by_author,
    add_features_to_df,
    format_raw_df,
)

REQUIRED_FEATURES = [
    "is_question",
    "action_verb_full",
    "language_question",
    "question_mark_full",
    "norm_text_len",
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
    return add_features_to_df(df.copy())


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


def test_feature_presence(df_with_features):
    for feat in REQUIRED_FEATURES:
        assert feat in df_with_features.columns


def test_feature_type(df_with_features):
    assert df_with_features["is_question"].dtype == bool
    assert df_with_features["action_verb_full"].dtype == bool
    assert df_with_features["language_question"].dtype == bool
    assert df_with_features["question_mark_full"].dtype == bool
    assert df_with_features["norm_text_len"].dtype == float
    assert df_with_features["vectors"].dtype == list


def test_vector_length(df_with_features):
    assert np.vstack(df_with_features["vectors"]).shape[1] == 300


def test_normalized_text_length(df_with_features):
    normalized_mean = df_with_features["norm_text_len"].mean()
    normalized_max = df_with_features["norm_text_len"].max()
    normalized_min = df_with_features["norm_text_len"].min()
    assert normalized_mean in pd.Interval(left=-1, right=1)
    assert normalized_max in pd.Interval(left=-1, right=1)
    assert normalized_min in pd.Interval(left=-1, right=1)
