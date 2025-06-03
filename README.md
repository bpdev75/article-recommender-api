# Hybrid Article Recommender API

This project provides a hybrid recommendation system to suggest articles to users, exposed through a lightweight HTTP API using Azure Functions.

The primary goal of this project is to **build and deploy** a functional recommender pipeline from scratch. Optimizing prediction quality is considered a future phase and is not the main focus at this stage.

---

## Recommender System Overview

### Dataset

We use the [News Portal User Interactions dataset from Globo.com (Kaggle)](https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom#clicks_sample.csv), which contains anonymized user interaction logs with news articles.

The pipeline leverages:

- **All hourly interaction files** (`clicks_hour_*.csv`) from the dataset.
  - These are loaded and concatenated to form a complete clickstream dataset.
- **Precomputed article embeddings** from the provided `article_embeddings.pickle` file.

### Approach

We implemented and compared three types of recommenders:

1. **Content-Based Filtering**
   - Article vectors are precomputed and stored in `article_embeddings.pickle`.
   - A **PCA (Principal Component Analysis)** was applied to reduce dimensionality.
   - Several dimensions were tested (50, 100, 150, 200, 250), with **dim=100** yielding the best performance.
   - User profiles are computed as the average of embeddings for articles the user has clicked on.
   - Recommendations are based on cosine similarity between the user vector and candidate articles.

2. **Collaborative Filtering**
   - We use implicit user feedback from the clickstream.
   - A matrix factorization approach (SVD from Surprise library) is used to predict user preferences per category.

3. **Hybrid Recommender**
   - Combines both models using a linear interpolation controlled by a parameter `alpha` (0 ≤ α ≤ 1).
   - The final score is a weighted average of content and collaborative predictions.

---

## Evaluation

Model performance was evaluated using **Hit Rate @5**, which measures how often the true clicked article is among the top 5 predicted.

| Model Type               | Hit Rate @5 |
|--------------------------|-------------|
| Content-Based (dim=100)  | 0.020       |
| Collaborative Filtering  | 0.015       |
| Hybrid (α = 0.5)         | 0.025       |

Note: model optimization was not the focus of this phase. The project emphasizes building an end-to-end deployment pipeline rather than achieving the highest possible predictive accuracy. Further optimization will follow.

---

## API Usage

The model is deployed as a serverless API using Azure Functions.

### Endpoint

`POST /predict`

### Request Body

```json
{
  "user_id": 123,
  "k": 5
}
```

### Response

```json
[81, 12, 44, 67, 105]
```
---

## Deployment

The API is deployed as an Azure Function (Python 3.11) with an HTTP trigger. The deployment is automated via GitHub, using Azure’s built-in CI/CD integration.
