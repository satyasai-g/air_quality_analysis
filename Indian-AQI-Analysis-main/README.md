# 🌍 Air Quality Prediction using Machine Learning

This project applies **machine learning techniques** to predict the **Air Quality Index (AQI)** across Indian cities.  
It combines **data preprocessing, exploratory data analysis (EDA), and regression modeling** to provide accurate AQI forecasts and insights into pollutant patterns.

## 🚀 Web App

A Streamlit web application is included for real-time AQI prediction.

### To run the web app locally:

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run app.py
   ```

The app will open in your browser, allowing you to input pollutant levels and get AQI predictions.

---

## 📂 Project Structure
- **`station_day.csv`** – Daily air quality data (pollutants, AQI, etc.)  
- **`stations.csv`** – Metadata about monitoring stations
- **`airquality (5).ipynb`** – Jupyter notebook with analysis and modeling
- **`train_models.py`** – Script to train the ML models
- **`app.py`** – Streamlit web app for predictions
- **`aqi_classifier.pkl`** – Trained classifier model
- **`aqi_regressor.pkl`** – Trained regressor model
- **`label_encoder.pkl`** – Label encoder for AQI buckets  
- **`airquality.ipynb`** – Jupyter Notebook with full workflow: preprocessing, EDA, modeling, evaluation  

---

## ⚙️ Methodology
1. **Data Cleaning & Preprocessing**
   - Handled missing values
   - Merged pollutant data with station metadata
   - Standardized features for modeling  

2. **Exploratory Data Analysis (EDA)**
   - Visualized pollutant trends across states  
   - Seasonal and temporal AQI patterns  
   - Correlations between pollutants and AQI  

3. **Modeling**
   - Applied regression for AQI prediction  
   - Evaluated performance with MAE, MSE, and R² metrics  

---

## 📊 Results
- **Mean Absolute Error (MAE):** 26.74  
- **Mean Squared Error (MSE):** 44.14  
- **R² Score:** 0.896 (~89.6%)  

✅ The model explains nearly **90% of variance** in AQI, with relatively low prediction error, making it effective for air quality forecasting.

---

## 🚀 Future Improvements
- Integrate meteorological data (temperature, humidity, wind speed)  
- Explore advanced models (Gradient Boosting, XGBoost, Deep Learning)  
- Deploy as a **real-time AQI dashboard** for public access  

---

## 🛠️ Installation & Usage
1. Clone the repository  
   ```bash
   git [clone https://github.com/yourusername/air-quality-prediction.git](https://github.com/Shadinbm/Indian-AQI-Analysis.git)
   cd air-quality-prediction
Install dependencies

bash
Copy code
pip install -r requirements.txt
Open the Jupyter Notebook

bash
Copy code
jupyter notebook airquality.ipynb
📌 License
This project is released under the MIT License – feel free to use and modify with attribution.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request if you’d like to improve the project.
