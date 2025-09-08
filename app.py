from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os

# Flask app
app = Flask(__name__)

# Load trained model
with open("Notebooks/model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Home page route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route (form)
@app.route('/predict', methods=['POST'])
def predict():
    prognosis_dict = {
        15:'Fungal infection', 4:'Allergy', 16:'GERD', 9:'Chronic cholestasis',
        14:'Drug Reaction', 33:'Peptic ulcer diseae', 1:'AIDS', 12:'Diabetes ',
        17:'Gastroenteritis', 6:'Bronchial Asthma', 23:'Hypertension ', 30:'Migraine',
        7:'Cervical spondylosis', 32:'Paralysis (brain hemorrhage)', 28:'Jaundice',
        29:'Malaria', 8:'Chicken pox', 11:'Dengue', 37:'Typhoid', 40:'Hepatitis A',
        19:'Hepatitis B', 20:'Hepatitis C', 21:'Hepatitis D', 22:'Hepatitis E',
        3:'Alcoholic hepatitis', 36:'Tuberculosis', 10:'Common Cold', 34:'Pneumonia',
        13:'Dimorphic hemmorhoids(piles)', 18:'Heart attack', 39:'Varicose veins',
        26:'Hypothyroidism', 24:'Hyperthyroidism', 25:'Hypoglycemia',
        31:'Osteoarthristis', 5:'Arthritis',
        0:'(vertigo) Paroymsal  Positional Vertigo', 2:'Acne',
        38:'Urinary tract infection', 35:'Psoriasis', 27:'Impetigo'
    }

    # Get symptoms from form
    data = request.form.to_dict()
    df = pd.DataFrame([data], columns=data.keys())
    df = df.astype(int)

    # Make prediction
    prediction = prognosis_dict[int(model.predict(df)[0])]

    return render_template("index.html", prediction_text=f"Disease: {prediction}")

# API route (JSON)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    content = request.json
    df = pd.DataFrame([content], columns=content.keys())
    df = df.astype(int)

    prediction = int(model.predict(df)[0])
    result = {
        15:'Fungal infection', 4:'Allergy', 16:'GERD', 9:'Chronic cholestasis',
        14:'Drug Reaction', 33:'Peptic ulcer diseae', 1:'AIDS', 12:'Diabetes ',
        17:'Gastroenteritis', 6:'Bronchial Asthma', 23:'Hypertension ', 30:'Migraine',
        7:'Cervical spondylosis', 32:'Paralysis (brain hemorrhage)', 28:'Jaundice',
        29:'Malaria', 8:'Chicken pox', 11:'Dengue', 37:'Typhoid', 40:'Hepatitis A',
        19:'Hepatitis B', 20:'Hepatitis C', 21:'Hepatitis D', 22:'Hepatitis E',
        3:'Alcoholic hepatitis', 36:'Tuberculosis', 10:'Common Cold', 34:'Pneumonia',
        13:'Dimorphic hemmorhoids(piles)', 18:'Heart attack', 39:'Varicose veins',
        26:'Hypothyroidism', 24:'Hyperthyroidism', 25:'Hypoglycemia',
        31:'Osteoarthristis', 5:'Arthritis',
        0:'(vertigo) Paroymsal  Positional Vertigo', 2:'Acne',
        38:'Urinary tract infection', 35:'Psoriasis', 27:'Impetigo'
    }[prediction]

    return jsonify({"prediction": result})

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
