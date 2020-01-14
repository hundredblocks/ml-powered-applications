import os
from pathlib import Path
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from ml_editor.data_processing import get_split_by_author

FEATURE_DISPLAY_NAMES = {
    "num_questions": "frequency of question marks",
    "num_periods": "frequency of periods",
    "num_commas": "frequency of commas",
    "num_exclam": "frequency of exclamation points",
    "num_quotes": "frequency of quotes",
    "num_colon": "frequency of colons",
    "num_semicolon": "frequency of semicolons",
    "num_stops": "frequency of stop words",
    "num_words": "word count",
    "num_chars": "word count",
    "num_diff_words": "vocabulary diversity",
    "avg_word_len": "vocabulary complexity",
    "polarity": "positivity of emotional sentiment",
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


def get_explainer():
    """
    Prepare LIME explainer using our training data. This is fast enough that
    we do not bother with serializing it
    :return: LIME explainer object
    """
    curr_path = Path(os.path.dirname(__file__))
    data_path = Path("../data/writers_with_features.csv")
    df = pd.read_csv(curr_path / data_path)
    train_df, test_df = get_split_by_author(df, test_size=0.2, random_state=40)
    explainer = LimeTabularExplainer(
        train_df[FEATURE_ARR].values,
        feature_names=FEATURE_ARR,
        class_names=["low", "high"],
    )
    return explainer


EXPLAINER = get_explainer()


def simplify_order_sign(order_sign):
    """
    Simplify signs to make display clearer for users
    :param order_sign: Input comparison operator
    :return: Simplifier operator
    """
    if order_sign in ["<=", "<"]:
        return "<"
    if order_sign in [">=", ">"]:
        return ">"
    return order_sign


def get_recommended_modification(simple_order, impact):
    """
    Generate a recommendation string from an operator and the type of impact
    :param simple_order: simplified operator
    :param impact: whether the change has positive or negative impact
    :return: formatted recommendation string
    """
    bigger_than_threshold = simple_order == ">"
    has_positive_impact = impact > 0

    if bigger_than_threshold and has_positive_impact:
        return "No need to decrease"
    if not bigger_than_threshold and not has_positive_impact:
        return "Increase"
    if bigger_than_threshold and not has_positive_impact:
        return "Decrease"
    if not bigger_than_threshold and has_positive_impact:
        return "No need to increase"


def parse_explanations(exp_list):
    """
    Parse explanations returned by LIME into a user readable format
    :param exp_list: explanations returned by LIME explainer
    :return: array of dictionaries containing user facing strings
    """
    parsed_exps = []
    for feat_bound, impact in exp_list:
        conditions = feat_bound.split(" ")

        # We ignore doubly bounded conditions , e.g. 1 <= a < 3 because
        # they are harder to formulate as a recommendation
        if len(conditions) == 3:
            feat_name, order, threshold = conditions

            simple_order = simplify_order_sign(order)
            recommended_mod = get_recommended_modification(simple_order, impact)

            parsed_exps.append(
                {
                    "feature": feat_name,
                    "feature_display_name": FEATURE_DISPLAY_NAMES[feat_name],
                    "order": simple_order,
                    "threshold": threshold,
                    "impact": impact,
                    "recommendation": recommended_mod,
                }
            )
    return parsed_exps


def get_recommendation_string_from_parsed_exps(exp_list):
    """
    Generate recommendation text we can display on a flask app
    :param exp_list: array of dictionaries containing explanations
    :return: HTML displayable recommendation text
    """
    recommendations = []
    for i, feature_exp in enumerate(exp_list):
        recommendation = "%s %s" % (
            feature_exp["recommendation"],
            feature_exp["feature_display_name"],
        )
        font_color = "green"
        if feature_exp["recommendation"] in ["Increase", "Decrease"]:
            font_color = "red"
        rec_str = """<font color="%s">%s) %s</font>""" % (
            font_color,
            i + 1,
            recommendation,
        )
        recommendations.append(rec_str)
    rec_string = "<br/>".join(recommendations)
    return rec_string
