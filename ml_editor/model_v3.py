import os
from pathlib import Path

import spacy
from sklearn.externals import joblib
from tqdm import tqdm
import pandas as pd
import nltk

from ml_editor.model_v2 import add_v2_text_features

nltk.download("vader_lexicon")

POS_NAMES = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

FEATURE_DISPLAY_NAMES = {
    "num_questions": "frequency of question marks",
    "num_periods": "frequency of periods",
    "num_commas": "frequency of commas",
    "num_exclam": "frequency of exclamation points",
    "num_quotes": "frequency of quotes",
    "num_colon": "frequency of colons",
    "num_semicolon": "frequency of semicolons",
    "num_stops": "frequency of stop words",
    "num_words": "question length",
    "num_chars": "question length",
    "num_diff_words": "vocabulary diversity",
    "avg_word_len": "vocabulary complexity",
    "polarity": "emotional sentiment",
    "ADJ": "frequency of adjectives",
    "ADP": "frequency of adpositions",
    "ADV": "frequency of adverbs",
    "AUX": "frequency of auxiliary verbs",
    "CONJ": "frequency of coordinating conjunctions",
    "DET": "frequency of determiners",
    "INTJ": "frequency of interjections",
    "NOUN": "frequency of nouns",
    "NUM": "frequency of numerals",
    "PART": "frequency of particles",
    "PRON": "frequency of pronouns",
    "PROPN": "frequency of proper nouns",
    "PUNCT": "frequency of punctuation",
    "SCONJ": "frequency of subordinating conjunctions",
    "SYM": "frequency of symbols",
    "VERB": "frequency of verbs",
    "X": "frequency of other words",
}


FEATURE_ARR = [
    "num_questions",
    "num_periods",
    "num_commas",
    "num_exclam",
    "num_quotes",
    "num_colon",
    "num_stops",
    "num_semicolon",
    "num_words",
    "num_chars",
    "num_diff_words",
    "avg_word_len",
    "polarity",
]
FEATURE_ARR.extend(POS_NAMES.keys())

SPACY_MODEL = spacy.load("en_core_web_sm")
tqdm.pandas()

curr_path = Path(os.path.dirname(__file__))

model_path = Path("../models/model_3.pkl")
MODEL = joblib.load(curr_path / model_path)


def get_features_from_input_text(text_input):
    arr_features = get_features_from_text_array([text_input])
    return arr_features.iloc[0]


def get_features_from_text_array(input_array):
    global FEATURE_ARR
    text_ser = pd.DataFrame(input_array, columns=["full_text"])
    text_ser = add_v2_text_features(text_ser.copy())
    features = text_ser[FEATURE_ARR].astype(float)
    return features


def get_model_probabilities_for_input_texts(text_array):
    global FEATURE_ARR, MODEL
    features = get_features_from_text_array(text_array)
    return MODEL.predict_proba(features)


def get_question_score_from_input(text):
    preds = get_model_probabilities_for_input_texts([text])
    positive_proba = preds[0][1]
    return positive_proba
