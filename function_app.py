import azure.functions as func
import logging
import json
import pickle
import pandas as pd
import requests
from io import BytesIO
from model.content_based import ContentBasedArticleRecommender

version = "2.0"
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
            response = requests.get("https://articlerecommendstorage.blob.core.windows.net/articlecontainer/articles_embeddings_50.pickle?sp=r&st=2025-06-17T17:33:34Z&se=2025-08-18T01:33:34Z&spr=https&sv=2024-11-04&sr=b&sig=4tdppluUbqMkGHs8i4q%2BkQxJfF%2BMrdKZm1toW%2BTf%2FZM%3D")
            response.raise_for_status()
            article_embeddings = pickle.load(BytesIO(response.content))

            train_df = pd.read_csv("https://articlerecommendstorage.blob.core.windows.net/articlecontainer/clicks_train.csv?sp=r&st=2025-06-17T17:34:50Z&se=2025-08-18T01:34:50Z&spr=https&sv=2024-11-04&sr=b&sig=hmjYdbAh1iXCEQic91FKVL0NFBANE6nJHcdXAIn4hYE%3D")
            test_df = pd.read_csv("https://articlerecommendstorage.blob.core.windows.net/articlecontainer/clicks_test.csv?sp=r&st=2025-06-17T17:34:31Z&se=2025-08-18T01:34:31Z&spr=https&sv=2024-11-04&sr=b&sig=P3C3cpJrmEZ41hT3hA4U%2FFKaEIi2vYUzQ%2BFz%2Bysznfs%3D")

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