import datetime

import dill
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier

from modules.encoding import encoder
from modules.load_data import load_data
from modules.outliers import outlier_remover


def pipeline() -> None:
    file_name = 'data/preprocessed.csv'

    df = load_data(file_name)
    df = outlier_remover(df)
    df_prepared = encoder(df)
    seed = 42

    X = df_prepared.drop('Attrition', axis=1)
    y = df_prepared['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=df_prepared['Attrition'], random_state=seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = CatBoostClassifier(depth=6,
                               random_state=seed,
                               iterations=2000,
                               logging_level='Silent',
                               auto_class_weights='Balanced'
                               )

    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    score = cross_val_score(model, X, y, cv=skf, scoring='accuracy', verbose=True)
    mean_score = score.mean()

    print(f'Mean accuracy {mean_score:.5} ')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Final Accuracy {acc: .5}')

    model_filename = f'Data/employe_atrition_prediction_model.pkl'
    with open(model_filename, 'wb') as file:
        dill.dump({
            'model': model,
            'metadata': {
                'Name': 'Employe Atrition Prediction Model',
                'Author': 'Sergey Jangozyan',
                'Version': 1.0,
                'Date': datetime.datetime.now(),
                'Type': 'CatboostClassifier',
                'Accuracy': acc,
                'Mean Accuracy': mean_score
            }
        }, file, recurse=True)

    print(f'\nModel is saved in pickle')


if __name__ == '__main__':
    pipeline()
