import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import pickle
import os

# Load dataset
data_path = os.path.join('data', 'chennai_cleaned.csv')
df = pd.read_csv(data_path)

# Drop unnecessary columns
df = df.drop(columns=['society'], errors='ignore')

# Features and target
X = df.drop("price(lakhs)", axis=1)
y = df["price(lakhs)"]

# Column types
categorical_cols = ['area_type', 'availability', 'location']
numerical_cols = ['size(bhk)', 'total_sqft', 'bath', 'balcony']

# Preprocessor pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# Final pipeline: preprocessing + model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save the entire pipeline (optional)
model_path = os.path.join('model', 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(pipeline.named_steps['model'], f)

# Save only the preprocessor
preprocessor_path = os.path.join('model', 'preprocessor.pkl')
with open(preprocessor_path, 'wb') as f:
    pickle.dump(pipeline.named_steps['preprocessor'], f)

print("✅ Model and preprocessor saved.")
