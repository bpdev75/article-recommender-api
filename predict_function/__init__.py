import azure.functions as func
import os
import json
import pickle
import numpy as np
import pandas as pd
from model.hybrid import HybridArticleRecommender

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "..", "data")

# Load article embeddings
with open(os.path.join(data_path, "articles_embeddings_50.pkl"), "rb") as f:
    article_embeddings = pickle.load(f)

# Load train and test dataframes
train_df = pd.read_csv(os.path.join(data_path, "clicks_train.csv"))
test_df = pd.read_csv(os.path.join(data_path, "clicks_test.csv"))

# Load the model
hybrid_model = HybridArticleRecommender(train_df, test_df, article_embeddings, alpha=0.5)

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        data = req.get_json()
        if "user_id" not in data:
            raise ValueError("The 'user_id' key is missing in the HTTP request")
        user_id = int(data["user_id"])
        if "k" not in data:
            raise ValueError("The 'k' key is missing in the HTTP request")
        k = int(data["k"])
        
        recommendations = hybrid_model.predict(user_id, k)
        return func.HttpResponse(
            json.dumps({"user_id": user_id, "recommendations": recommendations}),
            mimetype="application/json"
        )
    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)