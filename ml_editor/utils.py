from sklearn.ensemble import RandomForestClassifier


def get_filtering_model(classifier, features, labels):
    """
    Get prediction error for a binary classification dataset
    :param classifier: trained classifier
    :param features: input features
    :param labels: true labels
    """
    predictions = classifier.predict(features)
    # Create labels where errors are 1, and correct guesses are 0
    is_error = [pred != truth for pred, truth in zip(predictions, labels)]

    filtering_model = RandomForestClassifier()
    filtering_model.fit(features, is_error)
    return filtering_model
