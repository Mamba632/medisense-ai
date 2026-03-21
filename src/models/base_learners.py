from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import BASE_LEARNER_PARAMS


def get_base_learners() -> list[tuple[str, object]]:
    return [
        ("rf", RandomForestClassifier(**BASE_LEARNER_PARAMS["random_forest"])),
        ("xgb", XGBClassifier(**BASE_LEARNER_PARAMS["xgboost"])),
        ("lgbm", LGBMClassifier(**BASE_LEARNER_PARAMS["lightgbm"])),
        ("svm", SVC(**BASE_LEARNER_PARAMS["svm"])),
        ("knn", KNeighborsClassifier(**BASE_LEARNER_PARAMS["knn"])),
    ]
