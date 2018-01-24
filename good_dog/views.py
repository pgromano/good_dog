from . import preprocess

from good_dog import app
import flask
import numpy as np
import pandas as pd
from petfinder import PetfinderClient
import spacy

# Load language-model as global variable
NLP = spacy.load('en_core_web_lg')

@app.route('/',)
@app.route('/input', methods=['POST'])
def search_input():
    return flask.render_template("input.html")

@app.route('/output', methods=['POST'])
def search_output():
    # Get personality search terms
    print(flask.request.values)
    search = flask.request.form['search']
    gender = flask.request.form['gender']
    breed = flask.request.form['breed']
    age = flask.request.form['age']
    print(gender)
    print(breed)
    print(age)

    # Get authentication key and secret tokens
    with open('.auth_token.key', 'r') as f:
        key, secret = f.read().splitlines()

    # Initialize PetfinderClient to Query API
    # offset = np.random.randint(0, 250)
    pfclient = PetfinderClient(key, secret,
                                max_results=9, animal='dog', location='97403')

    # Prepare pandas dataframe
    columns = ["Age", "Breed", "Description",
               "Mix", "Name", "Sex", "Size",
               "Shelter_ID", "Address1",
               "Address2", "City", "Email",
               "Fax", "Phone", "State", "Zip", "Photos"]
    pet_database = pd.DataFrame(columns=columns)
    for pet in pfclient.find():
        values = []
        for col in columns:
            if col == 'Photos':
                found_photo = False
                for photo in pet['Photos']:
                    if photo['size'] == 'pn':
                        values.append(photo['url'])
                        found_photo = True
                        break
                if not found_photo:
                    values.append(None)
            else:
                values.append(pet[col])
        pet_database.loc[pet['Pet_ID']] = values

    # Remove empty descriptions
    n_entries = len(pet_database)
    print("Database originally has {:d} pet descriptions".format(n_entries))
    pet_database['Description'].replace('', np.nan, inplace=True)
    pet_database.dropna(subset=['Description'], inplace=True)
    print("Database has discarded {:d} entries".format(n_entries - len(pet_database)))

    # NOTE: Pet description must be copied, overwritten by preprocessing
    pet_database['Raw Description'] = pet_database['Description']

    # Preprocess pet-descriptions from query
    pet_database['Description'] = preprocess.preprocess(pet_database, NLP)
    pet_database['Topics'] = preprocess.get_topics(pet_database)
    pet_database['Score'] = preprocess.get_score(search, pet_database, NLP)
    pet_database.sort_values('Score', ascending=False, inplace=True)
    return flask.render_template("output.html", search=search, db=pet_database)
