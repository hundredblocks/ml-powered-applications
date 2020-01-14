import os
from pathlib import Path

import spacy
from sklearn.externals import joblib
from tqdm import tqdm
import pandas as pd
import nltk
from scipy.sparse import vstack, hstack

nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer


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

model_path = Path("../models/model_2.pkl")
vectorizer_path = Path("../models/vectorizer_2.pkl")
VECTORIZER = joblib.load(curr_path / vectorizer_path)
MODEL = joblib.load(curr_path / model_path)


def count_each_pos(df):
    """
    Count occurrences of each part of speech, and add it do an input DataFrame
    :param df: input DataFrame containing text that has been passed to SPACY_MODEL
    :return: DataFrame with occurrence counts
    """
    global POS_NAMES
    pos_list = df["spacy_text"].apply(lambda doc: [token.pos_ for token in doc])
    for pos_name in POS_NAMES.keys():
        df[pos_name] = (
            pos_list.apply(
                lambda x: len([match for match in x if match == pos_name])
            )
            / df["num_chars"]
        )
    return df


def get_word_stats(df):
    """
    Adds statistical features such as word counts to a DataFrame
    :param df: DataFrame containing a full_text column with training questions
    :return: DataFrame with new feature columns
    """
    global SPACY_MODEL
    df["spacy_text"] = df["full_text"].progress_apply(lambda x: SPACY_MODEL(x))

    df["num_words"] = (
        df["spacy_text"].apply(lambda x: 100 * len(x)) / df["num_chars"]
    )
    df["num_diff_words"] = df["spacy_text"].apply(lambda x: len(set(x)))
    df["avg_word_len"] = df["spacy_text"].apply(lambda x: get_avg_wd_len(x))
    df["num_stops"] = (
        df["spacy_text"].apply(
            lambda x: 100 * len([stop for stop in x if stop.is_stop])
        )
        / df["num_chars"]
    )

    df = count_each_pos(df.copy())
    return df


def get_avg_wd_len(tokens):
    """
    Returns average word length for a list of words
    :param tokens: array of words
    :return: average number of characters per word
    """
    if len(tokens) < 1:
        return 0
    lens = [len(x) for x in tokens]
    return float(sum(lens) / len(lens))


def add_char_count_features(df):
    """
    Adds counts of punctuation characters to a DataFrame
    :param df: DataFrame containing a full_text column with training questions
    :return: DataFrame with counts
    """
    df["num_chars"] = df["full_text"].str.len()

    df["num_questions"] = 100 * df["full_text"].str.count("\?") / df["num_chars"]
    df["num_periods"] = 100 * df["full_text"].str.count("\.") / df["num_chars"]
    df["num_commas"] = 100 * df["full_text"].str.count(",") / df["num_chars"]
    df["num_exclam"] = 100 * df["full_text"].str.count("!") / df["num_chars"]
    df["num_quotes"] = 100 * df["full_text"].str.count('"') / df["num_chars"]
    df["num_colon"] = 100 * df["full_text"].str.count(":") / df["num_chars"]
    df["num_semicolon"] = 100 * df["full_text"].str.count(";") / df["num_chars"]
    return df


def get_sentiment_score(df):
    """
    Uses nltk to return a polarity score for an input question
    :param df: DataFrame containing a full_text column with training questions
    :return: DataFrame with a polarity column
    """
    sid = SentimentIntensityAnalyzer()
    df["polarity"] = df["full_text"].progress_apply(
        lambda x: sid.polarity_scores(x)["pos"]
    )
    return df


def add_v2_text_features(df):
    """
    Adds multiple features used by the v2 model to a DataFrame
    :param df: DataFrame containing a full_text column with training questions
    :return: DataFrame with feature columns added
    """
    df = add_char_count_features(df.copy())
    df = get_word_stats(df.copy())
    df = get_sentiment_score(df.copy())
    return df


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
    text_ser = add_v2_text_features(text_ser.copy())
    vec_features = vstack(vectors)
    num_features = text_ser[FEATURE_ARR].astype(float)
    features = hstack([vec_features, num_features])
    return MODEL.predict_proba(features)


def get_question_score_from_input(text):
    """
    Helper method to get the probability for the positive class for one example question
    :param text: input string
    :return: estimated probability of question receiving a high score
    """
    preds = get_model_probabilities_for_input_texts([text])
    positive_proba = preds[0][1]
    return positive_proba


def get_pos_score_from_text(input_text):
    """
    Gets a score that can be displayed in the Flask app
    :param input_text: input string
    :return: estimated probability of question receiving a high score
    """
    positive_proba = get_question_score_from_input(input_text)
    output_str = (
        """
        Question score (0 is worst, 1 is best): 
        <br/>
        %s
    """
        % positive_proba
    )

    return output_str
