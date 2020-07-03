import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import pickle
import xgboost
from bankLoanValidator import initXGBoost, prediction

XG = xgboost.XGBClassifier()

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    initXGBoost(XG)
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    response = prediction(XG, final_features)
    flash('You bank loan has been {}'.format(response))
    return redirect(url_for('home'))



@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)