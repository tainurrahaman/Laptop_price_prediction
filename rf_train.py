import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data.csv")
print(df.head())

if 'Unnamed: 0' in df.columns:
  df.drop(columns=['Unnamed: 0','Unnamed: 0.1'],inplace=True)
df.columns

X = df.drop('price',axis=1)
y = df['price']

numerical_features = X.select_dtypes(include=['int64','float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

num_transformer = Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer,numerical_features),
        ('cat', cat_transformer,categorical_features)
    ]
)

rf_model = RandomForestRegressor(
  n_estimators= 448,
  max_depth= 20,
  min_samples_split=4,
  random_state= 42,
  n_jobs = -1   
)

rf_pipeline = Pipeline(
    [
        ('preprocessor',preprocessor),
        ('model', rf_model)
    ]
)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf_pipeline.fit(X_train,y_train)

y_pred = rf_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
r2 = r2_score(y_pred, y_test)

print(f"RMSE: {rmse: .4f}")
print(f"R2 Score: {r2: .4f}")

filename = "laptop_price_rf_model.pkl"

import pickle
with open(filename, "wb") as file:
  pickle.dump(rf_pipeline,file)

print("Random forest pipeline saved as Laptop_price_rf_model.pkl")