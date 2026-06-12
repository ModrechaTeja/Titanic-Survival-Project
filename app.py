from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    sex = request.form['Sex']
    sex_male = 1 if sex == 'male' else 0

    features = [[
        int(request.form['Pclass']),
        int(request.form['SibSp']),
        int(request.form['Parch']),
        float(request.form['Fare']),
        sex_male
    ]]

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "Passenger Survived"
    else:
        result = "Passenger Did Not Survive"

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)