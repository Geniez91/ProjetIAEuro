from fastapi import FastAPI, HTTPException
import pandas as pd
from typing import List
import pickle
import numpy as np
from transformers import pipeline
from sklearn.linear_model import LinearRegression

app = FastAPI(    
    title="Football Goals Prediction API",
    description="This API allows you to train a model on football match data and make predictions.",
    version="1.0")

# Load data from data.csv file
data = pd.read_csv("data.csv")

# Model to store the trained model
trained_model = None

@app.post("/train/", tags=["Training"], summary="Train the model", description="Train the model using the provided data.")
async def train_model():
    global data
    global trained_model
    if data is None:
        raise HTTPException(status_code=400, detail="Data not loaded.")
    
    # Specify columns to use for training
    columns_to_use = data[['player_id', 'player_club_id']]

    # Check if provided columns are present in the data
    for column in columns_to_use:
        if column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found in the data.")
    
    # Select numeric columns from the data
    X = data[columns_to_use].select_dtypes(include=['float64', 'int64']).drop(["goals"],axis=1)
    
    # Assuming 'goals' is a target column
    y = data['goals']
    
    # Example: training a simple linear regression model

    model = LinearRegression()
    model.fit(X, y)
    trained_model = model

    # Save model with pickle
    
    with open('models/model.pkl', 'wb') as model_file:
        pickle.dump(trained_model, model_file)
    
    return {"message": "Model trained successfully."}


@app.post("/predict/", tags=["Prediction"], summary="Make predictions", description="Make predictions using the trained model.")
async def predict(result: List[List[float]]):
    global trained_model

    # Load model with pickle
    with open('models/model.pkl', 'rb') as model_file:
        trained_model = pickle.load(model_file)


    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")
    # Your prediction logic here using trained_model
    print(type(result))
    print(result)
    predictions = trained_model.predict(np.array(result))
    predictions = np.clip(predictions, 0, None) 
    rounded_predictions = np.round(predictions)
    return {"predictions": str(rounded_predictions)}

# Documentation routes
@app.get("/", tags=["Root"], summary="Root endpoint", description="Welcome message for the API.")
async def read_root():
    return {"message": "Welcome to the football prediction API!"}

@app.get("/docs/", tags=["Documentation"], summary="API documentation", description="Documentation for the API.")
async def read_docs():
    return {"message": "API documentation"}

@app.get("/docs/train/", tags=["Documentation"], summary="Training documentation", description="Documentation for training the model.")
async def read_train_docs():
    return {"message": "Documentation for training the model"}

@app.get("/docs/predict/", tags=["Documentation"], summary="Prediction documentation", description="Documentation for making predictions.")
async def read_predict_docs():
    return {"message": "Documentation for making predictions"}

model = pipeline('text-classification', model='SamLowe/roberta-base-go_emotions')

@app.get("/model",tags=["HuggingFace"],description="Operation that calls the HuggingFace API to classify the emotion of a text.")
async def classify_emotion(text: str):
    emotion_prediction = model(text)[0]
    return {"emotion_prediction": emotion_prediction}
