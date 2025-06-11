import azure.functions as func
import logging
import os
import json
import pickle
import numpy as np
import pandas as pd
from model.hybrid import HybridArticleRecommender

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
version = "1.0"

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "data")

with open(os.path.join(data_path, "articles_embeddings_50.pickle"), "rb") as f:
        article_embeddings = pickle.load(f)

train_df = pd.read_csv(os.path.join(data_path, "clicks_train.csv"))
test_df = pd.read_csv(os.path.join(data_path, "clicks_test.csv"))
hybrid_model = HybridArticleRecommender(train_df, test_df, article_embeddings, alpha=0.5)

@app.route(route="predict_function")
def predict_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        data = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": "Invalid JSON format."
            }),
            status_code=400,
            mimetype="application/json"
        )

    if "user_id" not in data:
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": "Missing 'user_id' in request."
            }),
            status_code=400,
            mimetype="application/json"
        )

    if "k" not in data:
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": "Missing 'k' in request."
            }),
            status_code=400,
            mimetype="application/json"
        )
    
    try:
        user_id = int(data["user_id"])
        k = int(data["k"])
    except ValueError:
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": "'user_id' and 'k' must be integers."
            }),
            status_code=400,
            mimetype="application/json"
        )

    try:
        recommendations = hybrid_model.predict(user_id, k)
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": "Failed to generate recommendations."
            }),
            status_code=500,
            mimetype="application/json"
        )

    return func.HttpResponse(
        json.dumps({
            "status": "success",
            "data": {
                "recommendations": [int(r) for r in recommendations],
                "model_type": "hybrid",
                "version": version
            }
        }),
        status_code=200,
        mimetype="application/json"
    )
