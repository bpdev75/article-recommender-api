import pandas as pd
from model.base import AbstractArticleRecommender
from surprise import Dataset, Reader, SVD

class CollaborativeFilteringArticleRecommender(AbstractArticleRecommender):

    def __init__(self, train_df, test_df):
        super().__init__(train_df, test_df)
        train_data = self.build_surprise_dataset(train_df)
        test_data = self.build_surprise_dataset(test_df)
        self.trainset = train_data.build_full_trainset()
        self.testset = test_data.build_full_trainset().build_testset()
        self.model = SVD()
        self.model.fit(self.trainset)

    def _compute_category_based_ratings(self, clicks_df):
        """
        Compute a category-based rating for each (user_id, article_id) pair based on the proportion 
        of clicks the user made in each article category.

        The rating reflects user preference for a category, computed as:
            rating = (number of clicks by user in category) / (total number of clicks by user)

        Parameters
        ----------
        clicks_df : pandas.DataFrame
            A DataFrame with at least the following columns:
            - 'user_id': identifier for the user
            - 'article_id': identifier for the article
            - 'category_id': identifier for the category of the article

        Returns
        -------
        ratings_df : pandas.DataFrame
            A DataFrame with the columns:
            - 'user_id'
            - 'article_id'
            - 'category_id'
            - 'ratings': user preference score for the article based on category click behavior
        """
        # Count the number of clicks per (user, category)
        clicks_per_category = clicks_df.groupby(["user_id", "category_id"])["article_id"].count().reset_index(name="nb_clicks_per_category")

        # Total number of clicks per user
        total_clicks = clicks_per_category.groupby("user_id")["nb_clicks_per_category"].sum().reset_index(name="total_clicks")

        # Merge to compute rating
        rating_data = pd.merge(clicks_per_category, total_clicks, on="user_id", how="left")
        rating_data["ratings"] = rating_data["nb_clicks_per_category"] / rating_data["total_clicks"]

        # Drop intermediate columns
        rating_data.drop(columns=["nb_clicks_per_category", "total_clicks"], inplace=True)

        # Merge back to assign a rating to each (user, article) pair
        ratings_df = pd.merge(clicks_df, rating_data, on=["user_id", "category_id"], how="left")

        return ratings_df

    def build_surprise_dataset(self, clicks_df):
        ratings_df = self._compute_category_based_ratings(clicks_df)
        reader = Reader(rating_scale=(0, 1))
        return Dataset.load_from_df(ratings_df[['user_id', 'article_id', 'ratings']], reader)

    def predict_scores(self, user_id):
        predictions = {}
        for article_id in self.test_article_ids:
            pred = self.model.predict(user_id, article_id)
            predictions[pred.iid] = pred.est
        return predictions