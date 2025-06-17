import azure.functions as func
import logging
import os
import json
import pickle
import numpy as np
import pandas as pd
from model.content_based import ContentBasedArticleRecommender

version = "1.0"
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Global cache
model = None

@app.route(route="predict_function")
def predict_function(req: func.HttpRequest) -> func.HttpResponse:
    global model
    logging.info("Predict function triggered")

    if model is None:
        try:
            logging.info("Loading model and data...")
            base_path = os.path.dirname(__file__)
            data_path = os.path.join(base_path, "data")

            with open(os.path.join(data_path, "articles_embeddings_50.pickle"), "rb") as f:
                article_embeddings = pickle.load(f)

            train_df = pd.read_csv(os.path.join(data_path, "clicks_train.csv"))
            test_df = pd.read_csv(os.path.join(data_path, "clicks_test.csv"))

            model = ContentBasedArticleRecommender(train_df, test_df, article_embeddings)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "Server failed to load model."}),
                status_code=500,
                mimetype="application/json"
            )

    try:
        data = req.get_json()
    except ValueError:
        return func.HttpResponse(json.dumps({"status": "error", "message": "Invalid JSON format."}), status_code=400)

    if "user_id" not in data or "k" not in data:
        return func.HttpResponse(json.dumps({"status": "error", "message": "Missing 'user_id' or 'k'"}), status_code=400)

    try:
        user_id = int(data["user_id"])
        k = int(data["k"])
    except ValueError:
        return func.HttpResponse(json.dumps({"status": "error", "message": "'user_id' and 'k' must be integers."}), status_code=400)

    try:
        recommendations = model.predict(user_id, k)
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return func.HttpResponse(json.dumps({"status": "error", "message": "Failed to generate recommendations."}), status_code=500)

    return func.HttpResponse(
        json.dumps({
            "status": "success",
            "data": {
                "recommendations": [int(r) for r in recommendations],
                "model_type": model.name,
                "version": version
            }
        }),
        status_code=200,
        mimetype="application/json"
    )