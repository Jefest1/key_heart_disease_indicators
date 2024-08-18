# import libraries
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from flask import request, jsonify, Flask, render_template


# create a flask instance
app = Flask(__name__, static_folder='../static',
            template_folder='../templates')

# load preprocessor and models
model = tf.keras.models.load_model('./Models/model_4.keras')
preprocessor = joblib.load('./Models/preprocessor.pkl')

# Define feature names
feature_names = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Smoking',
                 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity',
                 'Asthma', 'KidneyDisease', 'SkinCancer', 'AgeCategory', 'Race', 'Diabetic', 'GenHealth']

# create the root endpoint


@app.route('/')
def home():
    return render_template('index.html')


# endpoint for templates
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    # Process form data into the correct format
    features = []
    for feature in feature_names:
        if feature in ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']:
            features.append(float(data[feature]))
        else:
            features.append(data[feature])

    df = pd.DataFrame([features], columns=feature_names)
    processed_features = preprocessor.fit_transform(df)
    prediction = model.predict(processed_features)
    result = np.round(tf.squeeze(prediction))
    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
