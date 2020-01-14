import os
from pathlib import Path

import pandas as pd
from sklearn.externals import joblib
from scipy.sparse import vstack, hstack

from ml_editor.data_processing import add_v1_features

FEATURE_ARR = [
    "action_verb_full",
    "question_mark_full",
    "text_len",
    "language_question",
]

curr_path = Path(os.path.dirname(__file__))

model_path = Path("../models/model_1.pkl")
vectorizer_path = Path("../models/vectorizer_1.pkl")
VECTORIZER = joblib.load(curr_path / vectorizer_path)
MODEL = joblib.load(curr_path / model_path)


def get_model_probabilities_for_input_texts(text_array):
    """
    Returns an array of probability scores representing
    the likelihood of a question receiving a high score
    format is: [ [prob_low_score1, prob_high_score_1], ... ]
    :param text_array: array of questions to be scored
    :return: array of predicted probabilities
    """
    global FEATURE_ARR, VECTORIZER, MODEL
    vectors = VECTORIZER.transform(text_array)
    text_ser = pd.DataFrame(text_array, columns=["full_text"])
    text_ser = add_v1_features(text_ser)
    vec_features = vstack(vectors)
    num_features = text_ser[FEATURE_ARR].astype(float)
    features = hstack([vec_features, num_features])
    return MODEL.predict_proba(features)


def get_model_predictions_for_input_texts(text_array):
    """
    Returns an array of labels for a given array of questions
    True represents high scores, False low scores
    format is: [ False, True, ...]
    :param text_array:  array of questions to be classified
    :return: array of classes
    """
    probs = get_model_probabilities_for_input_texts(text_array)
    predicted_classes = probs[:, 0] < probs[:, 1]
    return predicted_classes
