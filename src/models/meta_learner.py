from sklearn.linear_model import LogisticRegression

from src.config import STACKING_PARAMS, SEED


def get_meta_learner():
    return LogisticRegression(
        C=STACKING_PARAMS["meta_learner_C"],
        random_state=SEED,
        max_iter=1000,
    )
