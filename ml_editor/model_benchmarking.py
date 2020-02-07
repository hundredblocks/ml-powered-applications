import timeit
import numpy as np
import spacy
from keras_preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

text_input = [
    "Test sentence one",
    "Similarly, how fast does the model need to be?",
    "For some use cases, such as translating a short sentence, users will expect an answer immediately.",
    "For others, such as a medical diagnosis, patients would be happy to wait 24 hours if it meant that they would get the most accurate results.",
    "In our case, we will consider two potential ways we could deliver our product: through a submission box where the user writes, clicks submit and gets a result or by dynamically updating each time the user enters a new letter.",
    "While we may want to favor the latter because we would be able to make the tool much more interactive, we have to take into account that our models would then need to perform much faster."
    "A reasonable delay for a submission button can be up to five seconds, but for a model to run every few keystrokes, it would need to run significantly under a second. The most powerful models take longer to process data, so as we iterate through models, we will keep this tradeoff in mind.",
]

labels = np.array(range(len(text_input)))


class BenchmarkedModel:
    def __init__(self):
        pass

    def fit(self, data, labels):
        pass

    def predict(self, data):
        pass


class Count(BenchmarkedModel):
    def __init__(self):
        super().__init__()
        self.vectorizer = CountVectorizer()
        self.clf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced"
        )

    def fit(self, data, labels):
        self.clf.fit(self.vectorizer.fit_transform(data), labels)

    def predict(self, data):
        self.clf.predict(self.vectorizer.transform(data))


class GloVe(BenchmarkedModel):
    def __init__(self):
        super().__init__()
        self.clf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced"
        )
        self.vectorizer = spacy.load(
            "en_core_web_lg", disable=["parser", "tagger", "ner", "textcat"]
        )

    def fit(self, data, labels):
        spacy_emb = [self.vectorizer(x).vector for x in data]
        self.clf.fit(spacy_emb, labels)

    def predict(self, data):
        self.clf.predict([self.vectorizer(x).vector for x in data])


class DLModel(BenchmarkedModel):
    def __init__(self):
        super().__init__()
        max_features = 1024

        model = Sequential()
        model.add(Embedding(max_features, output_dim=256))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
        )
        self.clf = model
        self.vectorizer = Tokenizer()

    def fit(self, data, labels):
        self.vectorizer.fit_on_texts(data)
        processed_data = self.vectorizer.texts_to_matrix(data, mode="count")

        self.clf.fit(processed_data, labels, batch_size=16, epochs=10)

    def predict(self, data):
        processed_data = self.vectorizer.texts_to_matrix(data, mode="count")
        self.clf.predict(processed_data)


counts = Count()
counts.fit(text_input, labels)

glove = GloVe()
glove.fit(text_input, labels)

lstm = DLModel()
lstm.fit(text_input, labels)


def benchmark_inference(to_benchmark):
    """
    Run inference on a trained model
    :param to_benchmark: a model to benchmark
    """
    to_benchmark.predict(text_input)


if __name__ == "__main__":
    setup = """
from __main__ import benchmark_inference, counts, glove, lstm, text_input, labels
    """

    # We run inference multiple times on each model and take the fastest run
    # This helps reduce the impact of slowdowns due to other processes
    print("Timing count vectors (ms)")
    print(
        min(
            timeit.Timer("benchmark_inference(counts)", setup=setup).repeat(7, 1)
        )
    )
    print("Timing GloVe vectors (ms)")
    print(
        min(timeit.Timer("benchmark_inference(glove)", setup=setup).repeat(7, 1))
    )
    print("Timing DLModel vectors (ms)")
    print(
        min(timeit.Timer("benchmark_inference(lstm)", setup=setup).repeat(7, 1))
    )
