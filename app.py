from flask import Flask, render_template, redirect, url_for, flash, request
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from bson.objectid import ObjectId
from dotenv import load_dotenv
load_dotenv() 
import numpy as np
from rapidfuzz import process
app = Flask(__name__)
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import pickle
import tensorflow as tf
label_encoder_path = os.path.join("models", "label_encoder.pkl")
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

df = pd.read_csv("data/Training.csv")
all_columns = list(df.columns)
symptom_features = [col for col in all_columns if col.lower() != "prognosis"]
symptom_to_index = {symptom: idx for idx, symptom in enumerate(symptom_features)}

model_path = os.path.join("models", "model.h5")
model = tf.keras.models.load_model(model_path)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MONGO_URI'] = os.getenv('MONGO_URI')
mongo = PyMongo(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, _id, username, email, password):
        self.id = str(_id)
        self.username = username
        self.email = email
        self.password = password

    def is_active(self):
        return True

    def is_authenticated(self):
        return True

    def is_anonymous(self):
        return False

@login_manager.user_loader
def load_user(user_id):
    user_data = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if user_data:
        return User(user_data['_id'], user_data['username'], user_data['email'], user_data['password'])
    return None

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        existing_user = mongo.db.users.find_one({'email': email})
        if existing_user:
            flash("Email is already in use!", "danger")
            return redirect(url_for('signup'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = {"username": username, "email": email, "password": hashed_password}
        mongo.db.users.insert_one(new_user)
        flash("Account created successfully!", "success")
        return redirect(url_for('login'))
    return render_template("signup.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = mongo.db.users.find_one({'email': email})
        if user and bcrypt.check_password_hash(user['password'], password):
            user_obj = User(user['_id'], user['username'], user['email'], user['password'])
            login_user(user_obj)
            return redirect(url_for('home'))
        flash("Login failed. Check your credentials and try again.", "danger")
        return redirect(url_for('login'))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/period')
def period():
    return render_template('period.html')

@app.route('/symptom')
def symptom():
    return render_template('symptom.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/docs')
def docs():
    return render_template('doctor.html')


import re


def generate_ngrams(text, n=3):
    words = re.findall(r'\w+', text.lower())
    ngrams = []
    for i in range(len(words)):
        for j in range(i+1, min(i+n, len(words))+1):
            ngrams.append(" ".join(words[i:j]))
    return ngrams

def match_symptoms(user_input, symptoms_list):
    user_phrases = generate_ngrams(user_input, n=3)
    matched_symptoms = set()

    for phrase in user_phrases:
        result = process.extractOne(phrase, symptoms_list)
        if result and len(result) >= 2:
            best_match, score = result[0], result[1]
            if score>90:
                matched_symptoms.add(best_match)

    return list(matched_symptoms)


@app.route('/predict', methods = ['POST'])
def predict():
    user_input = request.form.get("symptoms","").strip()
    if not user_input:
        flash("Please enter your symptoms.", "warning")
        return redirect(url_for('symptom'))
    matched_symptoms = match_symptoms(user_input, symptom_features)
    input_vector = np.zeros((1,len(symptom_features)))
    for symptom in matched_symptoms:
        if symptom in symptom_to_index:
            input_vector[0, symptom_to_index[symptom]] = 1

    prediction = model.predict(input_vector)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_index])[0]
    return render_template("result.html", disease=predicted_disease, matched_symptoms=matched_symptoms)

if __name__ == "__main__":
    app.run(debug=True)
