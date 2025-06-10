import azure.functions as func
import os
import json
import pickle
import numpy as np
import pandas as pd
import logging
from model.hybrid import HybridArticleRecommender

logging.basicConfig(level=logging.INFO)

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "..", "data")

try:
    logging.info("Loading article embeddings...")
    with open(os.path.join(data_path, "articles_embeddings_50.pkl"), "rb") as f:
        article_embeddings = pickle.load(f)
    logging.info("Article embeddings loaded.")

    logging.info("Loading train dataframe...")
    train_df = pd.read_csv(os.path.join(data_path, "clicks_train.csv"))
    logging.info("Train dataframe loaded.")

    logging.info("Loading test dataframe...")
    test_df = pd.read_csv(os.path.join(data_path, "clicks_test.csv"))
    logging.info("Test dataframe loaded.")

    logging.info("Initializing HybridArticleRecommender model...")
    hybrid_model = HybridArticleRecommender(train_df, test_df, article_embeddings, alpha=0.5)
    logging.info("HybridArticleRecommender model initialized.")
except Exception as e:
    logging.error(f"An error occurred during model/data loading: {str(e)}", exc_info=True)
    raise e

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Received a new request to predict_function")

    try:
        data = req.get_json()
        logging.info(f"Request JSON: {data}")

        if "user_id" not in data:
            raise ValueError("The 'user_id' key is missing in the HTTP request")
        user_id = int(data["user_id"])
        logging.info(f"Parsed user_id: {user_id}")

        if "k" not in data:
            raise ValueError("The 'k' key is missing in the HTTP request")
        k = int(data["k"])
        logging.info(f"Parsed k: {k}")

        logging.info("Calling hybrid_model.predict()")
        recommendations = hybrid_model.predict(user_id, k)
        logging.info(f"Recommendations computed: {recommendations}")

        return func.HttpResponse(
            json.dumps({"user_id": user_id, "recommendations": recommendations}),
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)