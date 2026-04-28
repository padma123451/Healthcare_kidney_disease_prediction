
import numpy as np
from flask import Flask,request,jsonify
import mlflow.pyfunc

model = mlflow.pyfunc.load_model(
    "models:/kidney_disease_prediction/
Production"
)

# init the app
app = Flask(__name__)

@app.route("/predict", methods = ["POST"])
def predict():
    try:
        data = request.json["data"]
        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)
        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})
    

# run the app
if __name__ == "__main__":
    app.run(host ="0.0.0.0", port = 8000)