import pandas as pd
import numpy as np

from tqdm import tqdm

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score

from modules.load_data import load_data
from modules.encoding import encoder


file_name = 'data/preprocessed.csv'

df = load_data(file_name)
df_prepared = encoder(df)

# Split the dataset into train and test (80/20)
# Keep targets imbalance by using stratify attribute
X = df_prepared.drop('Attrition', axis=1)
y = df_prepared['Attrition']
seed = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=df_prepared['Attrition'], random_state=seed)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Dataset for results
results = pd.DataFrame(data={'Model': ['LogisticRegression', 'RandomForest', 'CatBoost', 'LightGBM'],
                             'Accuracy': np.nan,
                             'Precision': np.nan,
                             'Recall': np.nan,
                             'F1': np.nan,
                             'ROC-AUC': np.nan,
                             })


# Set the models
models = [
    {'name': 'LogisticRegression', 'model': LogisticRegression(
        random_state=seed, class_weight='balanced', n_jobs=-1)},
    {'name': 'RandomForest', 'model': RandomForestClassifier(
        random_state=seed, class_weight='balanced', n_jobs=-1)},
    {'name': 'CatBoost', 'model': CatBoostClassifier(
        random_state=seed, auto_class_weights='Balanced', logging_level='Silent')},
    {'name': 'LightGBM', 'model': LGBMClassifier(
        random_state=seed, class_weight='balanced', n_jobs=-1, verbose=-1)}
]


# Start training the models and record the results in the dataset
for n in tqdm(models, total=len(models)):
    clf = n['model']
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = float("%.4f" % accuracy_score(y_test, y_pred))
    pres = float("%.4f" % precision_score(y_test, y_pred))
    rec = float("%.4f" % recall_score(y_test, y_pred))
    f1 = float("%.4f" % f1_score(y_test, y_pred))
    roc_auc = float("%.4f" % roc_auc_score(y_test, y_prob))

    results.loc[results['Model'] == n['name'], [
        'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']] = [acc, pres, rec, f1, roc_auc]

# Display the results sorted by accuracy
print(results.sort_values(by=['Accuracy'], ascending=False))
