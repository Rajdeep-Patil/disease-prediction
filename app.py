from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load your trained model
with open("Notebooks/model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# List of all symptoms
symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 
            'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
            'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets',
            'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level',
            'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion',
            'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
            'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
            'acute_liver_failure', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
            'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain',
            'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
            'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
            'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
            'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
            'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
            'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
            'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
            'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
            'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
            'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
            'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
            'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
            'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
            'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
            'red_sore_around_nose', 'yellow_crust_ooze']

# Disease mapping
prognosis_dict = {
    15:'Fungal infection', 4:'Allergy', 16:'GERD', 9:'Chronic cholestasis', 14:'Drug Reaction',
    33:'Peptic ulcer diseae', 1:'AIDS', 12:'Diabetes', 17:'Gastroenteritis', 6:'Bronchial Asthma',
    23:'Hypertension', 30:'Migraine', 7:'Cervical spondylosis', 32:'Paralysis (brain hemorrhage)',
    28:'Jaundice', 29:'Malaria', 8:'Chicken pox', 11:'Dengue', 37:'Typhoid', 40:'Hepatitis A',
    19:'Hepatitis B', 20:'Hepatitis C', 21:'Hepatitis D', 22:'Hepatitis E', 3:'Alcoholic hepatitis',
    36:'Tuberculosis', 10:'Common Cold', 34:'Pneumonia', 13:'Dimorphic hemmorhoids(piles)',
    18:'Heart attack', 39:'Varicose veins', 26:'Hypothyroidism', 24:'Hyperthyroidism', 25:'Hypoglycemia',
    31:'Osteoarthristis', 5:'Arthritis', 0:'(vertigo) Paroymsal Positional Vertigo', 2:'Acne',
    38:'Urinary tract infection', 35:'Psoriasis', 27:'Impetigo'
}

# Home page -> form
@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms)

# Predict route -> show result on a separate page
@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    input_values = [int(form_data.get(symptom, 0)) for symptom in symptoms]
    
    prediction_code = int(model.predict([input_values])[0])
    prediction_name = prognosis_dict.get(prediction_code, "Unknown Disease")
    
    return render_template('result.html', prediction_text=f"Disease: {prediction_name}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
