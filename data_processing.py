import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit


def format_raw_df(df):
    """
    Cleanup data and join questions to answers
    :param df: raw DataFrame
    :return: processed DataFrame
    """
    # Fixing types and setting index
    df["PostTypeId"] = df["PostTypeId"].astype(int)
    df["Id"] = df["Id"].astype(int)
    df["AnswerCount"] = df["AnswerCount"].fillna(-1)
    df["AnswerCount"] = df["AnswerCount"].astype(int)
    df["OwnerUserId"].fillna(-1, inplace=True)
    df["OwnerUserId"] = df["OwnerUserId"].astype(int)
    df.set_index("Id", inplace=True, drop=False)

    df["is_question"] = df["PostTypeId"] == 1

    # Filtering out PostTypeIds other than documented ones
    df = df[df["PostTypeId"].isin([1, 2])]

    # Linking questions and answers
    df = df.join(
        df[
            ["Id", "Title", "body_text", "text_len", "Score", "AcceptedAnswerId"]
        ],
        on="ParentId",
        how="left",
        rsuffix="_question",
    )
    return df


def get_vectorized_representation(text_series, pretrained=False):
    """
    Generate a vectorized representation
    :param text_series: A pandas Series of text data
    :param pretrained: whether to use a pretrained model or vectorize from scratch
    :return: the vectorizer and a np array of dimension (len_series, embedding length)
    """
    if pretrained:
        vectorizer = spacy.load(
            "en_core_web_lg", disable=["parser", "tagger", "ner", "textcat"]
        )
        vectors = text_series.apply(lambda x: vectorizer(x).vector)

    else:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=5, max_features=2 ** 21
        )
        vectors = vectorizer.fit_transform(text_series)
    return vectorizer, vectors


def add_features_to_df(df):
    """
    Ads features to DataFrame
    :param df: DataFrame
    :return: DataFrame with additional features
    """
    df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")

    vectorizer, vectors = get_vectorized_representation(
        df["full_text"].copy(), pretrained=True
    )
    list_vectors = [list(vec) for vec in vectors]
    df["vectors"] = list_vectors

    df["action_verb_full"] = (
        df["full_text"].str.contains("can", regex=False)
        | df["full_text"].str.contains("What", regex=False)
        | df["full_text"].str.contains("should", regex=False)
    )
    df["language_question"] = (
        df["body_text"].str.contains("punctuate", regex=False)
        | df["body_text"].str.contains("capitalize", regex=False)
        | df["body_text"].str.contains("abbreviate", regex=False)
    )
    df["question_mark_full"] = df["full_text"].str.contains("?", regex=False)
    df["norm_text_len"] = get_normalized_series(df, "text_len")

    return df


def get_vectorized_inputs_and_label(df):
    """
    Concatenate DataFrame features with text vectors
    :param df: DataFrame with calculated features
    :param vectors: vectorized text
    :return: concatenated vector consisting of features and text
    """
    vectorized_features = np.append(
        np.vstack(df["vectors"]),
        df[
            [
                "action_verb_full",
                "question_mark_full",
                "norm_text_len",
                "language_question",
            ]
        ],
        1,
    )
    label = df["AcceptedAnswerId"].notna()

    return vectorized_features, label


def get_normalized_series(df, col):
    """
    Get a normalized version of a column
    :param df: DataFrame
    :param col: column name
    :return: normalized series
    """
    return (df[col] - df[col].mean()) / (df[col].max() - df[col].min())


def get_random_train_test_split(posts, test_size=0.3, random_state=40):
    """
    Get train/test split from DataFrame
    Assumes the DataFrame has one row per question example
    :param posts: all posts, with their labels
    :param test_size: the proportion to allocate to test
    :param random_state: a random seed
    """
    return train_test_split(
        posts, test_size=test_size, random_state=random_state
    )


def get_split_by_author(
    posts, author_id_column="OwnerUserId", test_size=0.3, random_state=40
):
    """
    Get train/test split
    Guarantee every author only appears in one of the splits
    :param posts: all posts, with their labels
    :param author_id_column: name of the column containing the author_id
    :param test_size: the proportion to allocate to test
    :param random_state: a random seed
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    splits = splitter.split(posts, groups=posts[author_id_column])
    train_idx, test_idx = next(splits)
    return posts.iloc[train_idx, :], posts.iloc[test_idx, :]
