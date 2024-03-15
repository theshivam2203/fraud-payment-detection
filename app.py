from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load('decision_tree_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = []
    features.append(int(request.form['type']))
    features.append(float(request.form['amount']))
    features.append(float(request.form['old_balance']))
    features.append(float(request.form['new_balance']))
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
