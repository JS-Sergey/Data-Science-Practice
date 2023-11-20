import pandas as pd
import numpy as np


def calculate_boundaries(data):
    """Function to calculate the IQR boundaries"""

    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    iqr = q75 - q25
    bounds = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

    return boundaries

def outlier_remover(df: pd.DataFrame) -> pd.DataFrame:
    """Function to replace the outliers with values close to the boundaries (inplace)"""

    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    numeric_data = df[numeric_cols].loc[:, df[numeric_cols].columns != 'Attrition']

    for col in numeric_data:
        boundaries = calculate_boundaries(df[col])
        is_outlier_0 = df[col] < boundaries[0]
        is_outlier_1 = df[col] > boundaries[1]
        if is_outlier_0.sum() != 0:
            df.loc[is_outlier_0, col] = int(boundaries[0] + 0.5)
        if is_outlier_1.sum() != 0:
            df.loc[is_outlier_1, col] = int(boundaries[1] + 0.5)

    return df
