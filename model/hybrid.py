from model.base import AbstractArticleRecommender
from model.content_based import ContentBasedArticleRecommender
from model.collaborative import CollaborativeFilteringArticleRecommender

class HybridArticleRecommender(AbstractArticleRecommender):
    def __init__(self, train_df, test_df, article_embeddings, alpha):
        super().__init__(train_df, test_df)
        self.cb_model = ContentBasedArticleRecommender(train_df, test_df, article_embeddings)
        self.cf_model = CollaborativeFilteringArticleRecommender(train_df, test_df)
        self.alpha = alpha

    def predict_scores(self, user_id):
        cb_preds = self.cb_model.predict_scores(user_id)
        cf_preds = self.cf_model.predict_scores(user_id)
        hybrid_preds = {}
        for article_id, cb_score in cb_preds.items():
            hybrid_score = self.alpha * cb_score + (1 - self.alpha) * cf_preds[article_id]
            hybrid_preds[article_id] = hybrid_score
        return hybrid_preds