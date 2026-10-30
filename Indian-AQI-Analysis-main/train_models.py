import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load data
print("Loading data...")
df1 = pd.read_csv("stations.csv")
df2 = pd.read_csv("station_day.csv")
print("Data loaded.")

# Merge
print("Merging...")
dfm = pd.merge(df1, df2, on='StationId', how='inner')
print("Merged.")

# Convert date
print("Converting date...")
dfm['Date'] = pd.to_datetime(dfm['Date'])
print("Date converted.")

# Data cleaning
print("Cleaning data...")
data = dfm.drop(columns=['PM10', 'NH3', 'Benzene', 'Toluene', 'Xylene'])
data = data.dropna()
print("Data cleaned.")

# Encode AQI_Bucket
print("Encoding...")
le = LabelEncoder()
data['AQI_Bucket'] = le.fit_transform(data['AQI_Bucket'])
print("Encoded.")

# Features for classification
fet_class = ['PM2.5', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3', 'AQI']
x_class = data[fet_class]
y_class = data['AQI_Bucket']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(x_class, y_class, test_size=0.2, random_state=42)

# Train classifier
print("Training classifier...")
rfclass = RandomForestClassifier(n_estimators=100, random_state=42)
rfclass.fit(X_train_class, y_train_class)
print("Classifier trained.")

# Features for regression
fet_reg = ['PM2.5', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3']
x_reg = data[fet_reg]
y_reg = data['AQI']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(x_reg, y_reg, test_size=0.2, random_state=42)

# Train regressor
print("Training regressor...")
rfreg = RandomForestRegressor(n_estimators=100, random_state=42)
rfreg.fit(X_train_reg, y_train_reg)
print("Regressor trained.")

# Save models
print("Saving models...")
joblib.dump(rfclass, 'aqi_classifier.pkl')
joblib.dump(rfreg, 'aqi_regressor.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Models saved.")