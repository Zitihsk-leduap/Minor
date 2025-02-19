from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")

## Initialize FastAPI
app = FastAPI()


class InputData(BaseModel):
  features:list[float]

@app.post('/predict')
def predict(data:InputData):
  X_input = np.array(data.features).reshape(1,-1)

  # Making predictions
  prediction = model.predict(X_input)[0]

  if prediction==0:
    return "No"
  else:
    return "Yes"

  # return {"Prediction":int(prediction)}