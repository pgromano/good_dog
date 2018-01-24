import spacy
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# Set global parameters
N_TOPICS = 5
STOP_WORDS = set(stopwords.words('english'))

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
    clean_document = nltk.word_tokenize(document)

    # Remove abstraction
    clean_document = [word.lower() for word in clean_document if word.lower() not in STOP_WORDS]
    clean_document = [word for word in clean_document if word not in PUNCTUATIONS]

    # Lemmatize
    lemma = WordNetLemmatizer()
    clean_document = " ".join([lemma.lemmatize(word) for word in clean_document])
    return clean_document


def _filter(document, NLP):
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
    for sentence in NLP(document).doc.sents:
        if SENT_CLF.predict([sentence.text]) == 'D':
            filtered_document += sentence.text
    return filtered_document


def _get_topics(document):
    document = document.split()
    n_features = len(document)

    # Fit tfidf vectorizer
    tf_vec = TfidfVectorizer(max_features=int(n_features / 2), stop_words='english').fit(document)

    # Convert document to tfidf
    tf = tf_vec.transform(document)

    # Get feature names
    feature_names = tf_vec.get_feature_names()

    # Fit NMF topic model
    model = NMF(alpha=0.1, l1_ratio=0.75).fit(tf)

    # Set up lemmatizer
    lemma = WordNetLemmatizer()

    # Output topics
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        for i in topic.argsort()[:-N_TOPICS - 1:-1]:
            topics.append(lemma.lemmatize(feature_names[i]))
    return " ".join(np.unique(topics))


def _get_score(document, search, NLP):
    return 100 * NLP(search).similarity(NLP(document))


def preprocess(pet_database, NLP):
    """ Filter and clean words """
    pet_database['Description'] = pet_database['Description'].apply(_filter, args=(NLP, ))
    pet_database['Description'] = pet_database['Description'].apply(_clean)
    return pet_database['Description']


def get_topics(pet_database):
    return pet_database['Description'].apply(_get_topics)


def get_score(search, pet_database, NLP):
    """ Cosine similarity of search and pet-description word vectors """
    return pet_database['Topics'].apply(_get_score, args=(search, NLP))
