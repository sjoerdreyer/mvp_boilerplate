from fastapi import FastAPI
import pickle
from package_folder.utils import from_number_to_flower


app = FastAPI()

@app.get("/")
def root():
    return {'greeting':"hello"}

@app.get("/predict")
def predict(sepal_length=1,
            sepal_width=1,
            petal_length=1,
            petal_width=1):
    with open('models/best_model.pkl', 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

    pretty_prediction = from_number_to_flower(float(prediction[0]))

    return {"prediction": pretty_prediction}
