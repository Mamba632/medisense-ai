import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from src.config import STACKING_PARAMS, SEED


class StackingEnsemble:
    def __init__(self, base_learners, meta_learner, n_folds=STACKING_PARAMS["n_folds"]):
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.n_folds = n_folds
        self.fitted_base_learners_ = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        n_learners = len(self.base_learners)
        oof_predictions = np.zeros((n_samples, n_learners))

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)

        for i, (name, learner) in enumerate(self.base_learners):
            for train_idx, val_idx in skf.split(X, y):
                cloned = clone(learner)
                cloned.fit(X[train_idx], y[train_idx])
                oof_predictions[val_idx, i] = cloned.predict_proba(X[val_idx])[:, 1]

        # Retrain each base learner on full training set
        self.fitted_base_learners_ = []
        for name, learner in self.base_learners:
            fitted = clone(learner)
            fitted.fit(X, y)
            self.fitted_base_learners_.append((name, fitted))

        # Train meta-learner on OOF predictions
        self.meta_learner.fit(oof_predictions, y)
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack(
            [
                learner.predict_proba(X)[:, 1]
                for _, learner in self.fitted_base_learners_
            ]
        )
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
