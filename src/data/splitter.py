import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED, VAL_RATIO, TEST_RATIO


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_test_ratio = VAL_RATIO + TEST_RATIO
    test_fraction_of_remainder = TEST_RATIO / val_test_ratio

    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=SEED,
        stratify=df[target_col],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_fraction_of_remainder,
        random_state=SEED,
        stratify=temp_df[target_col],
    )

    return train_df, val_df, test_df
