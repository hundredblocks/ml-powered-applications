import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from scipy.sparse import vstack, hstack


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
        df[["Id", "Title", "body_text", "Score", "AcceptedAnswerId"]],
        on="ParentId",
        how="left",
        rsuffix="_question",
    )
    return df


def train_vectorizer(df):
    """
    Train a vectorizer for some data.
    Returns the vectorizer to be used to transform non-training data, in
    addition to the training vectors
    :param df: data to use to train the vectorizer
    :return: trained vectorizers and training vectors
    """
    vectorizer = TfidfVectorizer(
        strip_accents="ascii", min_df=5, max_df=0.5, max_features=10000
    )

    vectorizer.fit(df["full_text"].copy())
    return vectorizer


def get_vectorized_series(text_series, vectorizer):
    """
    Vectorizes an input series using a pre-trained vectorizer
    :param text_series: pandas Series of text
    :param vectorizer: pretrained sklearn vectorizer
    :return: array of vectorized features
    """
    vectors = vectorizer.transform(text_series)
    vectorized_series = [vectors[i] for i in range(vectors.shape[0])]
    return vectorized_series


def add_text_features_to_df(df):
    """
    Ads features to DataFrame
    :param df: DataFrame
    :param pretrained_vectors: whether to use pretrained vectors for embeddings
    :return: DataFrame with additional features
    """
    df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")
    df = add_v1_features(df.copy())

    return df


def add_v1_features(df):
    """
    Add our first features to an input DataFrame
    :param df: DataFrame of questions
    :return: DataFrame with added feature columns
    """
    df["action_verb_full"] = (
        df["full_text"].str.contains("can", regex=False)
        | df["full_text"].str.contains("What", regex=False)
        | df["full_text"].str.contains("should", regex=False)
    )
    df["language_question"] = (
        df["full_text"].str.contains("punctuate", regex=False)
        | df["full_text"].str.contains("capitalize", regex=False)
        | df["full_text"].str.contains("abbreviate", regex=False)
    )
    df["question_mark_full"] = df["full_text"].str.contains("?", regex=False)
    df["text_len"] = df["full_text"].str.len()
    return df


def get_vectorized_inputs_and_label(df):
    """
    Concatenate DataFrame features with text vectors
    :param df: DataFrame with calculated features
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
    label = df["Score"] > df["Score"].median()

    return vectorized_features, label


def get_feature_vector_and_label(df, feature_names):
    """
    Generate input and output vectors using the vectors feature and
     the given feature names
    :param df: input DataFrame
    :param feature_names: names of feature columns (other than vectors)
    :return: feature array and label array
    """
    vec_features = vstack(df["vectors"])
    num_features = df[feature_names].astype(float)
    features = hstack([vec_features, num_features])
    labels = df["Score"] > df["Score"].median()
    return features, labels


def get_normalized_series(df, col):
    """
    Get a normalized version of a column
    :param df: DataFrame
    :param col: column name
    :return: normalized series using Z-score
    """
    return (df[col] - df[col].mean()) / df[col].std()


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
