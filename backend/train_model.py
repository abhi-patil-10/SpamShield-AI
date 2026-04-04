import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle 

#load the dataset
df = pd.read_csv("C:\\Users\\Prime\\Desktop\\Digital_twin_health_system\\data\\heart_cleveland_upload.csv")
print(df.head())

#Split feature and target
X = df.drop("condition" , axis=1)
y = df["condition"]

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Train the model
model = RandomForestClassifier()
model.fit(X_train,y_train)

#Accuracy
accuracy = model.score(X_test,y_test)
print(f"Model Accuracy : {accuracy}")

#Save the model
with open("C:\\Users\\Prime\\Desktop\\Digital_twin_health_system\\models\\model.pkl" , "wb") as file:
    pickle.dump(model,file)

print("Model saved successfully as model.pkl")


#model testing with a sample data point

# sample = X.iloc[0].values.reshape(1, -1)
# prediction = model.predict_proba(sample)

# print("Sample Risk:", prediction[0][1])