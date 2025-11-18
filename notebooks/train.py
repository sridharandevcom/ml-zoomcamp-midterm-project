import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# ---------------------------
# Load and clean data
# ---------------------------
df = pd.read_csv("../data/Titanic-Dataset.csv")

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# ---------------------------
# Train/Val Split
# ---------------------------
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train.Survived.values
y_val = df_val.Survived.values

train_dicts = df_train.drop("Survived", axis=1).to_dict(orient="records")
val_dicts = df_val.drop("Survived", axis=1).to_dict(orient="records")

dv = DictVectorizer(sparse=True)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# ---------------------------
# Final Model Training (XGBoost)
# ---------------------------
params = {
    'eta': 0.1,
    'max_depth': 4,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

model = xgb.train(params, dtrain, num_boost_round=200)

# ---------------------------
# Save Model + DV
# ---------------------------
with open("xgb_model.bin", "wb") as f:
    pickle.dump((dv, model), f)

print("Model saved as xgb_model.bin")
