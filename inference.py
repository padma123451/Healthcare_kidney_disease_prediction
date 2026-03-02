import joblib
import numpy as np

model = joblib.load("classifier.pkl")
scaler = joblib.load("scaler.pkl")
data = [1]*24
data = np.array(data).reshape(1,-1)
data_scaled = scaler.transform(data)
res = model.predict(data_scaled)
print(res)