import joblib
from flask import Flask, render_template, request
import google.generativeai as genai

app = Flask(__name__)

import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini = genai.GenerativeModel("gemini-pro")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Predict')
def prediction():
    return render_template('index.html')

@app.route('/form', methods=["POST"])
def brain():
    Nitrogen = float(request.form['Nitrogen'])
    Phosphorus = float(request.form['Phosphorus'])
    Potassium = float(request.form['Potassium'])
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    Ph = float(request.form['ph'])
    Rainfall = float(request.form['Rainfall'])

    values = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]

    if 0 < Ph <= 14 and 0 < Temperature < 100 and 0 < Humidity:
        try:
            with open('crop_app.joblib', 'rb') as file:
                model = joblib.load(file)
            arr = [values]
            acc = model.predict(arr)[0]
            tips = get_crop_tips(acc)
            tips=tips.split("+")
            for i in tips:
                if len(i)<1:
                    tips.remove(i)
            
            return render_template('prediction.html', prediction=str(acc),tips=tips)
        except Exception as e:
            return f"Error loading the model: {str(e)}"
    else:
        return "Sorry... Error in entered values in the form. Please check the values and fill it again"
    

def get_crop_tips(crop):
    # Provide a descriptive and specific prompt
    prompt = f'''
    Provide tips for growing {crop} effectively, considering the local climate and soil conditions and seperate each tips with + symbol'''

    # Make a request to the Gemini Pro API with the prompt
    response = gemini.generate_content(prompt)
    return response.text

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
