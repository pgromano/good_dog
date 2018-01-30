import pandas as pd
import numpy as np
import string

import nltk
from nltk.stem import PorterStemmer

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer

# Set global parameters
N_TOPICS = 10
LANGUAGE = "english"
SENTENCES_COUNT = 2
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words(LANGUAGE))
except:
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words(LANGUAGE))

# Ignore adoption words
STOP_WORDS.add('foster')
STOP_WORDS.add('adopt')
STOP_WORDS.add('breeder')
STOP_WORDS.add('house')
STOP_WORDS.add('say')
STOP_WORDS.add('way')
STOP_WORDS.add('use')

# Ignore US States
states_abr = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

states = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
         'Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho',
         'Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
         'Maine' 'Maryland','Massachusetts','Michigan','Minnesota',
         'Mississippi', 'Missouri','Montana','Nebraska','Nevada',
         'New Hampshire','New Jersey','New Mexico','New York',
         'North Carolina','North Dakota','Ohio',
         'Oklahoma','Oregon','Pennsylvania','Rhode Island',
         'South  Carolina','South Dakota','Tennessee','Texas','Utah',
         'Vermont','Virginia','Washington','West Virginia',
         'Wisconsin','Wyoming']

for state_abr, state in zip(states_abr, states):
    STOP_WORDS.add(state_abr)
    STOP_WORDS.add(state)

# Remove dog breeds
with open('good_dog/static/models/dog_breeds', 'r') as f:
    dog_breeds = f.read().splitlines()

for breed in dog_breeds:
    STOP_WORDS.add(breed)

PUNCTUATIONS = set(string.punctuation)
SENT_CLF = joblib.load('good_dog/static/models/nb_sentence_classifier.pkl')


def _clean(document):
    # Tokenize
    words = nltk.word_tokenize(document)

    # Remove abstraction
    clean_document = [word.lower() for word in words if word.lower() not in STOP_WORDS]
    clean_document = [word for word in clean_document if word not in PUNCTUATIONS]

    # Stem
    stemmer = PorterStemmer()
    clean_document = " ".join([stemmer.stem(word) for word in clean_document])
    return clean_document


def _filter(document):
    """ Tool to remove junk aspects from the pet document

    Arguments
    ---------
    document : str
        Document of describing pet.

    Returns
    -------
    filtered_document : str
        Concatenated sentence strings that the NB classifier identifies
        as a "dog document".
    """
    filtered_document = " "
    for sentence in nltk.sent_tokenize(document):
        if SENT_CLF.predict([sentence]) == 'D':
            filtered_document += sentence
    return filtered_document


def _get_topics(document):
    document = document.split()
    if len(document) >= 2:
        n_features = len(document)
    else:
        n_features = 1

    # Fit tfidf vectorizer
    tf_vec = TfidfVectorizer(max_features=n_features, stop_words='english').fit(document)

    # Convert document to tfidf
    tf = tf_vec.transform(document)

    # Get feature names
    feature_names = tf_vec.get_feature_names()

    # Fit NMF topic model
    model = NMF(alpha=0.1, l1_ratio=0.75).fit(tf)

    # Output topics
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        for i in topic.argsort()[:-N_TOPICS - 1:-1]:
            topics.append(feature_names[i])
    return " ".join(np.unique(topics))


def _get_summary(document):
    parser = PlaintextParser.from_string(document, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = STOP_WORDS

    summary = " "
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        summary += " ".join(sentence.words)

    return summary


def _get_score(document, search, NLP):
    return 100 * NLP(search).similarity(NLP(document))


def preprocess(pet_database):
    """ Filter and clean words """

    # Preprocessing
    pet_database['Description'] = pet_database['Description'].apply(_filter)
    pet_database['Description'] = pet_database['Description'].apply(_clean)

    # Delete any empty descriptions
    pet_database['Description'].replace('', np.nan, inplace=True)
    pet_database.dropna(subset=['Description'], inplace=True)
    return pet_database['Description']


def get_topics(pet_database):
    return pet_database['Summary'].apply(_get_topics)


def get_summary(pet_database):
    return pet_database['Description'].apply(_get_summary)


def get_score(search, pet_database, NLP):
    """ Cosine similarity of search and pet-description word vectors """
    return pet_database['Summary'].apply(_get_score, args=(search, NLP))
