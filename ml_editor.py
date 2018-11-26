import argparse
import logging
import sys

import pyphen
import nltk

pyphen.language_fallback('en_US')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_out = logging.StreamHandler(sys.stdout)
console_out.setLevel(logging.DEBUG)
logger.addHandler(console_out)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Receive text to be edited"
    )
    parser.add_argument(
        'text',
        metavar='input text',
        type=str
    )
    args = parser.parse_args()
    return args.text


def clean_input(text):
    # To keep things simple at the start, let's only keep ASCII characters
    return str(text.encode().decode('ascii', errors='ignore'))


def preprocess_input(text):
    tokens = nltk.word_tokenize(text)
    return tokens


def compute_flesch_reading_ease(total_syllables, total_words, total_sentences):
    return 206.85 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)


def compute_average_word_length(tokens):
    word_lengths = [len(word) for word in tokens]
    return sum(word_lengths) / len(word_lengths)


def compute_unique_words_fraction(tokens):
    unique_words = len(set([word for word in tokens]))
    total_words = len([word for word in tokens])
    return unique_words / total_words


def count_word_usage(tokens, word_list):
    return len([word for word in tokens if word.lower() in word_list])


def count_word_syllables(word):
    dic = pyphen.Pyphen(lang='en_US')
    # this returns our word, with hyphens ("-") inserted in between each syllable
    hyphenated = dic.inserted(word)
    return len(hyphenated.split("-"))


def count_sentence_syllables(tokens):
    # Our tokenizer leaves punctuation as a separate word, so we filter for it here
    punctuation = ".,!?/"
    return sum([count_word_syllables(word) for word in tokens if word not in punctuation])


def get_sentence_lenth(sentence_tokens):
    punctuation = ".,!?/"
    return len([word for word in sentence_tokens if word not in punctuation])


def compute_sentence_length(text):
    # TODO split sentences
    pass


def get_suggestions(tokens):
    told_said_usage = count_word_usage(tokens, ['told', 'said'])
    but_and_usage = count_word_usage(tokens, ['but', 'and'])
    wh_adverbs = count_word_usage(tokens, ['when', 'where', 'why', 'whence', 'whereby', 'wherein', 'whereupon'])
    average_word_length = compute_average_word_length(tokens)
    unique_words_fraction = compute_unique_words_fraction(tokens)

    # TODO add sentences logic
    syllables = count_sentence_syllables(tokens)
    sentence_length = get_sentence_lenth(tokens)
    flesch_score = compute_flesch_reading_ease(syllables, sentence_length, 1)
    logger.info("Adverb usage: %s told/said, %s but/and, %s wh adverbs" % (told_said_usage, but_and_usage, wh_adverbs))
    logger.info(
        "Average word length %.2f, fraction of unique words %.2f" % (average_word_length, unique_words_fraction))
    logger.info("%d syllables, %.2f flesch score" % (syllables, flesch_score))


if __name__ == '__main__':
    input_text = parse_arguments()
    processed = clean_input(input_text)
    tokenized = preprocess_input(processed)
    suggestions = get_suggestions(tokenized)
