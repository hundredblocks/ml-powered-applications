"""inference.py: This module contains function stubs serving as book examples.
The functions are not used for the ml_editor app or notebook
"""

from functools import lru_cache

REQUIRED_FEATURES = [
    "is_question",
    "action_verb_full",
    "language_question",
    "question_mark_full",
    "norm_text_len",
]


def find_absent_features(data):
    missing = []
    for feat in REQUIRED_FEATURES:
        if feat not in data.keys():
            missing.append(feat)
    return missing


def check_feature_types(data):
    types = {
        "is_question": bool,
        "action_verb_full": bool,
        "language_question": bool,
        "question_mark_full": bool,
        "norm_text_len": float,
    }
    mistypes = []
    for field, data_type in types:
        if not isinstance(data[field], data_type):
            mistypes.append((data[field], data_type))
    return mistypes


def run_heuristic(question_len):
    pass


@lru_cache(maxsize=128)
def run_model(question_data):
    """
    This is a stub function. We actually use the lru_cache with a purpose
    in app.py
    :param question_data:
    """
    # Insert any slow model inference below
    pass


def validate_and_handle_request(question_data):
    missing = find_absent_features(question_data)
    if len(missing) > 0:
        raise ValueError("Missing feature(s) %s" % missing)

    wrong_types = check_feature_types(question_data)
    if len(wrong_types) > 0:
        # If data is wrong but we have the length of the question, run heuristic
        if "text_len" in question_data.keys():
            if isinstance(question_data["text_len"], float):
                return run_heuristic(question_data["text_len"])
        raise ValueError("Incorrect type(s) %s" % wrong_types)

    return run_model(question_data)


def verify_output_type_and_range(output):
    if not isinstance(output, float):
        raise ValueError("Wrong output type %s, %s" % (output, type(output)))
    if not 0 < output < 1:
        raise ValueError("Output out of range %s, %s" % output)


def validate_and_correct_output(question_data, model_output):
    # Verify type and range and raise errors accordingly
    try:
        # Raises value error if model output is incorrect
        verify_output_type_and_range(model_output)
    except ValueError:
        # We run a heuristic, but could run a different model here
        run_heuristic(question_data["text_len"])

    # If we did not raise an error, we return our model result
    return model_output
