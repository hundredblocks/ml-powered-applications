import os
from pathlib import Path

import spacy
from sklearn.externals import joblib
from tqdm import tqdm
import pandas as pd
import nltk

from ml_editor.explanation_generation import (
    parse_explanations,
    get_recommendation_string_from_parsed_exps,
    EXPLAINER,
    FEATURE_ARR,
)
from ml_editor.model_v2 import add_v2_text_features

nltk.download("vader_lexicon")

SPACY_MODEL = spacy.load("en_core_web_sm")
tqdm.pandas()

curr_path = Path(os.path.dirname(__file__))

model_path = Path("../models/model_3.pkl")
MODEL = joblib.load(curr_path / model_path)


def get_features_from_input_text(text_input):
    """
    Generates features for a unique text input
    :param text_input: question string
    :return: one row series containing v3 model features
    """
    arr_features = get_features_from_text_array([text_input])
    return arr_features.iloc[0]


def get_features_from_text_array(input_array):
    """
    Generated features for an input array of text
    :param input_array: array of input questions
    :return: DataFrame of features
    """
    text_ser = pd.DataFrame(input_array, columns=["full_text"])
    text_ser = add_v2_text_features(text_ser.copy())
    features = text_ser[FEATURE_ARR].astype(float)
    return features


def get_model_probabilities_for_input_texts(text_array):
    """
    Returns estimated v3 model probabilities from input text array
    :param text_array: array of input questions
    :return: array of predictions
    """
    global MODEL
    features = get_features_from_text_array(text_array)
    return MODEL.predict_proba(features)


def get_question_score_from_input(text):
    """
    Returns v3 model probability for a unique text input
    :param text: input string
    :return: estimated probability of question receiving a high score
    """
    preds = get_model_probabilities_for_input_texts([text])
    positive_proba = preds[0][1]
    return positive_proba


def get_recommendation_and_prediction_from_text(input_text, num_feats=10):
    """
    Gets a score and recommendations that can be displayed in the Flask app
    :param input_text: input string
    :param num_feats: number of features to suggest recommendations for
    :return: current score along with recommendations
    """
    global MODEL
    feats = get_features_from_input_text(input_text)

    pos_score = MODEL.predict_proba([feats])[0][1]
    print("explaining")
    exp = EXPLAINER.explain_instance(
        feats, MODEL.predict_proba, num_features=num_feats, labels=(1,)
    )
    print("explaining done")
    parsed_exps = parse_explanations(exp.as_list())
    recs = get_recommendation_string_from_parsed_exps(parsed_exps)
    output_str = """
    Current score (0 is worst, 1 is best):
     <br/>
     %s
    <br/>
    <br/>
    
    Recommendations (ordered by importance):
    <br/>
    <br/>
    %s
    """ % (
        pos_score,
        recs,
    )
    return output_str
