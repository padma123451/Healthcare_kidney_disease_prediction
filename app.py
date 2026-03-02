import joblib
import numpy as np
from flask import Flask, render_template,request

model = joblib.load("classifier.pkl")
scaler = joblib.load("scaler.pkl")

# init the app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["POST"])
def predict():
    try:
        data = [float(value) for value in request.form.values()]
        data = np.array(data).reshape(1, -1)
        data_scaled =scaler.transform(data)
        prediction = model.predict(data_scaled)
        return f"Prediction: {prediction[0]}"
    except Exception as e:
        return f"Error: {e}"
   

@app.route("/contact")
def contact():
    return "welcome to contact page"


# run the app
if __name__ == "__main__":
    app.run(debug = True)