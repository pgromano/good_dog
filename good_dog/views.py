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
def homepage():
    return flask.render_template("homepage.html")

@app.route('/skills',)
def skills():
    return flask.render_template("skills.html")

@app.route('/education',)
def education():
    return flask.render_template("education.html")

@app.route('/good_dog', methods=['GET', 'POST'])
def good_dog_search():
    return flask.render_template("input.html")

@app.route('/output', methods=['POST'])
def good_dog_results():
    # Get personality search terms
    search = flask.request.form['search']
    location = flask.request.form['location']
    gender = flask.request.form['gender']
    breed = flask.request.form['breed']
    age = flask.request.form['age']
    size = flask.request.form['size']
    print("Looking for a {:s}, {:s} {:s} dog of {:s}-age in {:s}.".format(size, gender, breed, age, location))

    # Check Location
    if location == "":
        location = 78705

    # Check Gender
    if gender == "Gender":
        gender = None
    elif gender == "Female":
        gender = "F"
    elif gender == "Male":
        gender = "M"

    # Check Breed
    if breed == "Breed":
        breed = None

    # Check Age
    if age == "Age":
        age = None
    elif age == "Puppy":
        age = "Baby"

    # Check Size
    if size == "Size":
        size = None

    # Get authentication key and secret tokens
    with open('.auth_token.key', 'r') as f:
        key, secret = f.read().splitlines()

    query = {'animal': 'dog',
             'max_results': 50,
             'age': age,
             'breed': breed,
             'location': location,
             'sex': gender,
             'size': size}

    # Initialize PetfinderClient to Query API
    # offset = np.random.randint(0, 250)
    pfclient = PetfinderClient(key, secret, **query)

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
    pet_database['Description'] = preprocess.preprocess(pet_database)

    # Process pet descriptions
    if len(pet_database) > 8:
        pet_database['Summary'] = preprocess.get_summary(pet_database)
        pet_database['Topics'] = preprocess.get_topics(pet_database)
        pet_database['Score'] = preprocess.get_score(search, pet_database, NLP)
        pet_database.sort_values('Score', ascending=False, inplace=True)
        return flask.render_template("output.html", search=search, db=pet_database.iloc[:9], location=str(location))
    else:
        return flask.render_template("too_few_dogs.html", search=search, location=str(location))
