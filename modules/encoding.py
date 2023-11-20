import pandas as pd


def encoder(df: pd.DataFrame) -> pd.DataFrame:
    """Function to encode the categorical features and add them to the dataset"""

    categorical_cols = list(df.select_dtypes(include='O').columns)

    encoded = pd.get_dummies(df[categorical_cols], dtype='int8')
    df = df.drop(categorical_cols, axis=1)
    df_prepared = pd.concat([df, encoded], axis=1)

    return df_prepared
