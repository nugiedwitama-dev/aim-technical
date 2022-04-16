from flask import Flask, render_template, request
import joblib
import numpy as np
model = joblib.load("model/model1.pkl")
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    data_features = dict(request.form).values()
    data_features = np.array([float(x) for x in data_features])
  
    pred = model.predict([data_features])
    return render_template('index.html', result=pred)

if __name__ == '__main__':
    app.run(port=5000, debug=True)