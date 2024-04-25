from flask import Flask, render_template, request
import pickle
import xgboost

app = Flask(__name__)

with open('Temperature_prediction_model.pkl', 'rb') as file:
    temperature_model = pickle.load(file)

with open('rainfall_prediction_model.pkl', 'rb') as file:
    rainfall_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('page.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/h')
def h():
    return render_template('h.html')

@app.route("/predict_temperature")
def predict_temperature():
    return redirect("http://localhost:8503")

@app.route('/predict_rain')
def predict_rain():
    return redirect("http://localhost:8504")

@app.route('/predict_final')
def predict_rainfall():
    return redirect("http://localhost:8502")

if __name__ == '__main__':
    app.run(debug=True)
