import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class AbstractArticleRecommender(ABC):
    """
    An abstract class to recommend articles for a given user id
    """

    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.test_article_ids = test_df["article_id"].unique().astype(int)

    def predict(self, user_id, k=5):
        """
        Return the top-k recommended article IDs for a given user.
        
        Parameters
        ----------
        user_id : int
            ID of the user for whom to make the recommendation.
        k : int, optional
            Number of top articles to recommend (default is 5).
        
        Returns
        -------
        List[int]
            List of recommended article IDs.
        """
        pred_scores = self.predict_scores(user_id)
        pred_scores = dict(sorted(pred_scores.items(), key=lambda item: item[1], reverse=True)[:k])
        return [article_id for article_id, _ in pred_scores.items()]
    
    @abstractmethod
    def predict_scores(self, user_id):
        """
        Return a dictionary {article_id: recommendation_score} for the given user
        
        Parameters
        ----------
        user_id : int
            ID of the user for whom to make the recommendation.
        
        Returns
        -------
        dict[int, float]
            List of recommended article IDs with their recommendation scores.
        """
        pass

    def hit_rate_at_k(self, k=5, sample_users=200):
        """
        Compute the Hit Rate at K for the collaborative filtering model.

        The Hit Rate at K measures how often the ground truth (i.e., articles a user 
        actually interacted with in the test set) appears among the top-K recommended 
        articles for that user. It is defined as the proportion of users for whom at 
        least one of the top-K predicted articles matches an article in their test set.

        Parameters
        ----------
        k : int, optional (default=5)
            The number of top articles to consider in the recommendation list.

        sample_users : int
            Number of users to sample from the test set (to limit computation).
        
        Returns
        -------
        float
            The average hit rate across all users in the test set. A value of 1.0 
            means every user had at least one relevant article among the top-K recommendations,
            while 0.0 means no user had a relevant recommendation.
        """
        # Convert test_df into user -> set of read article_ids using groupby
        uid_2_read_articles = self.test_df.groupby("user_id")["article_id"].apply(set)

        # Sample users using numpy (faster than random.sample for arrays)
        all_uids = uid_2_read_articles.index.values
        if len(all_uids) == 0:
            return 0.0

        sampled_uids = np.random.choice(all_uids, size=min(sample_users, len(all_uids)), replace=False)

        hits = 0
        for uid in sampled_uids:
            recommended = self.predict(uid, k)  # top-k recommendations
            ground_truth = uid_2_read_articles[uid]

            # Use set intersection for fast lookup
            if any(article in ground_truth for article in recommended):
                hits += 1

        return hits / len(sampled_uids)