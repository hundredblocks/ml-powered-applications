import sys
import os
from pathlib import Path
import pandas as pd

import pytest

# Needed for pytest to resolve imports properly
from sklearn.externals import joblib

from ml_editor.data_ingestion import parse_xml_to_csv
from ml_editor.data_processing import (
    format_raw_df,
    add_text_features_to_df,
    get_vectorized_series,
    get_feature_vector_and_label,
)
from ml_editor.model_v1 import get_model_predictions_for_input_texts

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

CURR_PATH = Path(os.path.dirname(__file__))
XML_PATH = Path("fixtures/MiniPosts.xml")
CSV_PATH = Path("fixtures/MiniPosts.csv")
MODEL_PATH = Path("../models/model_1.pkl")
VECTORIZER_PATH = Path("../models/vectorizer_1.pkl")

FEATURE_NAMES = [
    "action_verb_full",
    "question_mark_full",
    "text_len",
    "language_question",
]

# Make sure we have a csv
@pytest.fixture(scope="session", autouse=True)
def get_csv():
    parse_xml_to_csv(CURR_PATH / XML_PATH, save_path=CURR_PATH / CSV_PATH)


@pytest.fixture
def df_with_features():
    df = pd.read_csv(CURR_PATH / CSV_PATH)
    df = format_raw_df(df.copy())
    return add_text_features_to_df(df.copy())


@pytest.fixture
def trained_v1_model():
    model_path = Path(CURR_PATH / MODEL_PATH)
    clf = joblib.load(model_path)
    return clf


@pytest.fixture
def trained_v1_vectorizer():
    vectorizer_path = Path(CURR_PATH / VECTORIZER_PATH)
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer


def test_model_prediction_dimensions(
    df_with_features, trained_v1_vectorizer, trained_v1_model
):
    df_with_features["vectors"] = get_vectorized_series(
        df_with_features["full_text"].copy(), trained_v1_vectorizer
    )

    features, labels = get_feature_vector_and_label(
        df_with_features, FEATURE_NAMES
    )

    probas = trained_v1_model.predict_proba(features)
    # the model makes one prediction per input example
    assert probas.shape[0] == features.shape[0]
    # the model predicts probabilities for two classes
    assert probas.shape[1] == 2


def test_model_proba_values(
    df_with_features, trained_v1_vectorizer, trained_v1_model
):
    df_with_features["vectors"] = get_vectorized_series(
        df_with_features["full_text"].copy(), trained_v1_vectorizer
    )

    features, labels = get_feature_vector_and_label(
        df_with_features, FEATURE_NAMES
    )

    probas = trained_v1_model.predict_proba(features)
    # the model's probabilities are between 0 and 1
    assert (probas >= 0).all() and (probas <= 1).all()


def test_model_predicts_no_on_bad_question():
    input_text = "This isn't even a question. We should score it poorly"
    is_question_good = get_model_predictions_for_input_texts([input_text])
    # The model classifies the question as poor
    assert not is_question_good[0]
