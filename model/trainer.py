import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/chennai_cleaned.csv")
X = df.drop("price", axis=1)
y = df["price"]

# Preprocessing
categorical_cols = ["area_type", "availability", "location"]
numerical_cols = ["size", "total_sqft", "bath", "balcony"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
], remainder='passthrough')

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

# Save model & preprocessor
pickle.dump(pipeline.named_steps['model'], open("model/model.pkl", "wb"))
pickle.dump(preprocessor, open("model/preprocessor.pkl", "wb"))
