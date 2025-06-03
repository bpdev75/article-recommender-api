from base import AbstractArticleRecommender
import numpy as np

class ContentBasedArticleRecommender(AbstractArticleRecommender):
    """
    A class to recommend articles to users based on cosine similarity between user and article embeddings.
    
    This recommender uses precomputed article embeddings and infers user embeddings by averaging 
    the embeddings of articles each user has interacted with. It supports filtering out previously 
    read articles and retrieving top-k recommendations.
    
    Parameters
    ----------
    all_clicks_df : pandas.DataFrame
        DataFrame containing at least 'user_id' and 'article_id' columns representing click history.
    article_embeddings : numpy.ndarray
        Numpy array of shape (nb_articles, embedding_dim) containing article embeddings.
    """

    def __init__(self, train_df, test_df, article_embeddings):
        super().__init__(train_df, test_df)
        self.article_embeddings = article_embeddings
        self.nb_articles = article_embeddings.shape[0]
        self.embedding_dim = article_embeddings.shape[1]

        # Build user embeddings
        self.user_embeddings = self._compute_user_embeddings()

    def _compute_user_embeddings(self):
        """
        Compute user embeddings by averaging the embeddings of articles each user has interacted with.
        """
        grouped = self.train_df.groupby("user_id")["article_id"].apply(list)
        user_ids = grouped.index
        user_embeddings_matrix = []

        for article_ids in grouped:
            user_embeddings_matrix.append(np.mean(self.article_embeddings[article_ids], axis=0))

        user_embeddings_array = np.vstack(user_embeddings_matrix)
        return dict(zip(user_ids, user_embeddings_array))

    def predict_scores(self, user_id):
        user_vector = self.user_embeddings[user_id].reshape(1, -1)

        # Compute similarity with all articles
        similarities = cosine_similarity(user_vector, self.article_embeddings[self.test_article_ids])[0]
        
        return dict(zip(self.test_article_ids, similarities))

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))