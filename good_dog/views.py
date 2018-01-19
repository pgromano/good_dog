from . import preprocess

from good_dog import app
import flask
import pandas as pd
from petfinder import PetfinderClient
import spacy

# Load language-model as global variable
NLP = spacy.load('en_core_web_lg')

@app.route('/')
@app.route('/input')
def search_input():
    return flask.render_template("input.html")

@app.route('/output')
def search_output():
    # Get personality search terms
    search = flask.request.args.get('search')

    # Get authentication key and secret tokens
    with open('.auth_token.key', 'r') as f:
        key, secret = f.read().splitlines()

    # Initialize PetfinderClient to Query API
    pfclient = PetfinderClient(key, secret, max_results=9, animal='dog', location='97403')

    # Prepare pandas dataframe
    columns = ["Age", "Breed", "Description",
               "Mix", "Name", "Sex", "Size",
               "Shelter_ID", "Address1",
               "Address2", "City", "Email",
               "Fax", "Phone", "State", "Zip", "Image_URL"]
    pet_database = pd.DataFrame(columns=columns)
    for pet in pfclient.find():
        pet_database.loc[pet["Pet_ID"]] = [pet[col] for col in columns]

    # NOTE: Pet description must be copied, overwritten by preprocessing
    pet_database['Raw Description'] = pet_database['Description']

    # Preprocess pet-descriptions from query
    pet_database['Description'] = preprocess.preprocess(pet_database, NLP)
    pet_database['Topics'] = preprocess.get_topics(pet_database)
    pet_database['Score'] = preprocess.get_score(search, pet_database, NLP)
    pet_database.sort_values('Score', ascending=False, inplace=True)
    return flask.render_template("output.html", search=search, db=pet_database)
