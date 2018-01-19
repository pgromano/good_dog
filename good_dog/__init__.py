from . import preprocess
from flask import Flask

app = Flask(__name__)
from good_dog import views
