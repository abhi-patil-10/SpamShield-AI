from flask import Flask, request,jsonify , render_template
import pickle
import numpy as np
import json

app = Flask(__name__)

with open("C:\\Users\\Prime\\Desktop\\Digital_twin_health_system\\models\\model.pkl", "rb") as file:
    model = pickle.load(file)
    
@app.route("/" , methods=["GET","POST"])
def home():
    
    return render_template("base.html")

@app.route("/predict" , methods=["GET","POST"])
def predict():
    data = None
    prediction  = 0 
    health_score = 0
    if request.method == "POST":
        data = request.form.get("json_input")
        json_data = json.loads(data)
        
         # Convert input to array
        features = np.array(list(json_data.values())).reshape(1, -1)

        # Prediction
        prediction = model.predict_proba(features)[0][1]
        health_score = (1 - prediction) * 100
        
        print(json_data)
    return render_template("predict.html", data=data, prediction=(prediction * 100), health_score=health_score)


@app.route("/simulation" , methods=["GET","POST"])
def simulation():
    current_risk = 0
    new_risk = 0
    health_score_before = 0
    health_score_after = 0
    improvement = 0


    if request.method == "POST":
        updated_data = request.form.get("updated_data")
        updated_json_data = json.loads(updated_data)
        
        current = updated_json_data["current"]
        changes = updated_json_data["changes"]
        
        simulated = current.copy()
        simulated.update(changes)
        
        current_features = np.array(list(current.values())).reshape(1, -1)
        
        new_features = np.array(list(simulated.values())).reshape(1, -1)
        
        current_risk = model.predict_proba(current_features)[0][1]
        new_risk = model.predict_proba(new_features)[0][1]
        
        health_score_before = 100 - (current_risk * 100)
        health_score_after = 100 - (new_risk * 100)
        improvement = float(current_risk - new_risk)


    
    return render_template("simulation.html", current_risk=float(current_risk), new_risk=float(new_risk) , health_score_before=float(health_score_before), health_score_after=float(health_score_after), improvement=improvement)

if __name__ == "__main__":
    app.run(debug=True)

