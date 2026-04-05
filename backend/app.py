from flask import Flask, request,jsonify , render_template
import pickle
import numpy as np
import json

app = Flask(__name__)

with open("C:\\Users\\Prime\\Desktop\\Digital_twin_health_system\\models\\model.pkl", "rb") as file:
    model = pickle.load(file)
    
@app.route("/" , methods=["GET","POST"])
def home():
    
    return render_template("index.html")

@app.route("/predict" , methods=["GET","POST"])
def predict():
    data = None
    prediction  = None  
    if request.method == "POST":
        data = request.form.get("data")
        json_data = json.loads(data)
        
         # Convert input to array
        features = np.array(list(json_data.values())).reshape(1, -1)

        # Prediction
        prediction = model.predict_proba(features)[0][1]
        
        print(json_data)
    return render_template("predict.html", data=data, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

