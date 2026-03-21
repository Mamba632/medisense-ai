import joblib

from src.config import DATASETS, MODELS_DIR, SEED
from src.data.loader import load_raw_dataset, get_target_column
from src.data.preprocessor import MediSensePreprocessor
from src.data.splitter import stratified_split
from src.models.base_learners import get_base_learners
from src.models.dl_model import KerasClassifierWrapper
from src.models.stacking import StackingEnsemble
from src.models.meta_learner import get_meta_learner
from src.evaluation.metrics import compute_metrics, format_metrics


def train_pipeline(dataset_name: str, use_dl: bool = True) -> dict:
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\n{'=' * 60}")
    print(f"Training pipeline for: {dataset_name}")
    print(f"{'=' * 60}")

    # Load data
    df = load_raw_dataset(dataset_name)
    target_col = get_target_column(dataset_name)

    # Handle liver dataset target (1/2 -> 1/0)
    if dataset_name == "liver":
        df[target_col] = df[target_col].map({1: 1, 2: 0})

    # Split
    train_df, val_df, test_df = stratified_split(df, target_col)
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Preprocess
    preprocessor = MediSensePreprocessor(dataset_name)
    X_train = preprocessor.fit_transform(train_df.drop(columns=[target_col]))
    X_val = preprocessor.transform(val_df.drop(columns=[target_col]))
    X_test = preprocessor.transform(test_df.drop(columns=[target_col]))

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    # SMOTE on training set
    try:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=SEED)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train.shape[0]} training samples")
    except ImportError:
        print("imbalanced-learn not installed, skipping SMOTE")

    # Base learners
    base_learners = get_base_learners()

    if use_dl:
        dl_wrapper = KerasClassifierWrapper(input_dim=X_train.shape[1])
        base_learners.append(("keras_mlp", dl_wrapper))

    # Train individual base learners and report
    print("\nBase learner results on validation set:")
    for name, learner in base_learners:
        from sklearn.base import clone

        fitted = clone(learner)
        fitted.fit(X_train, y_train)
        y_pred = fitted.predict(X_val)
        y_prob = fitted.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_prob)
        print(f"\n  {name}:")
        print(format_metrics(metrics))

    # Stacking ensemble
    print("\nTraining stacking ensemble...")
    meta_learner = get_meta_learner()
    ensemble = StackingEnsemble(base_learners, meta_learner)
    ensemble.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_test = ensemble.predict(X_test)
    y_prob_test = ensemble.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_pred_test, y_prob_test)

    print("\nStacking Ensemble — Test Set Results:")
    print(format_metrics(test_metrics))

    # Save artifacts
    save_dir = MODELS_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(ensemble, save_dir / "stacking_ensemble.pkl")
    joblib.dump(preprocessor, save_dir / "preprocessor.pkl")
    print(f"\nModels saved to {save_dir}")

    return test_metrics
