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
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens


def compute_flesch_reading_ease(total_syllables, total_words, total_sentences):
    return 206.85 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)


def compute_average_word_length(tokens):
    word_lengths = [len(word) for word in tokens]
    return sum(word_lengths) / len(word_lengths)


def compute_total_average_word_length(sentence_list):
    lengths = [compute_average_word_length(tokens) for tokens in sentence_list]
    return sum(lengths) / len(lengths)


def compute_total_unique_words_fraction(sentence_list):
    all_words = [word for word_list in sentence_list for word in word_list]
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)


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


def count_total_syllables(sentence_list):
    return sum([count_sentence_syllables(sentence) for sentence in sentence_list])


def count_words_per_sentence(sentence_tokens):
    punctuation = ".,!?/"
    return len([word for word in sentence_tokens if word not in punctuation])


def count_total_words(sentence_list):
    return sum([count_words_per_sentence(sentence) for sentence in sentence_list])


def get_suggestions(sentence_list):
    told_said_usage = sum([count_word_usage(tokens, ['told', 'said']) for tokens in sentence_list])
    but_and_usage = sum([count_word_usage(tokens, ['but', 'and']) for tokens in sentence_list])
    wh_adverbs_usage = sum(
        [count_word_usage(tokens, ['when', 'where', 'why', 'whence', 'whereby', 'wherein', 'whereupon']) for tokens in
         sentence_list])
    logger.info(
        "Adverb usage: %s told/said, %s but/and, %s wh adverbs" % (told_said_usage, but_and_usage, wh_adverbs_usage))
    average_word_length = compute_total_average_word_length(sentence_list)
    unique_words_fraction = compute_total_unique_words_fraction(sentence_list)

    logger.info(
        "Average word length %.2f, fraction of unique words %.2f" % (average_word_length, unique_words_fraction))

    number_of_syllables = count_total_syllables(sentence_list)
    number_of_words = count_total_words(sentence_list)
    number_of_sentences = len(sentence_list)
    logger.info(
        "%d syllables, %d words, %d sentences" % (number_of_syllables, number_of_words, number_of_sentences))

    flesch_score = compute_flesch_reading_ease(number_of_syllables, number_of_words, number_of_sentences)
    logger.info("%d syllables, %.2f flesch score" % (number_of_syllables, flesch_score))


if __name__ == '__main__':
    input_text = parse_arguments()
    processed = clean_input(input_text)
    tokenized_sentences = preprocess_input(processed)
    suggestions = get_suggestions(tokenized_sentences)
