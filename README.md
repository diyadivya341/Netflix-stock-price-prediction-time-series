# Netflix-stock-price-prediction-time-series


📌 Project Summary: 

This project aims to analyze and forecast Netflix stock prices using time series forecasting techniques. It provides an interactive Streamlit web application where users can visualize stock trends, perform decomposition, check stationarity, and generate price forecasts using models like ARIMA and Simple Exponential Smoothing.

The model was implemented using ARIMA, with optional support for Exponential Smoothing. Users can choose forecasting duration and test set size dynamically within the app.

📂 Dataset Details

The dataset used contains historical Netflix stock data with daily prices. Key features include:

Close (Target variable – daily closing price)

Date (as datetime index)

The data was preprocessed and saved using joblib for fast loading in the web app.

🛠️ Technologies Used

Python Libraries:

Streamlit (for building the web interface)

Pandas, NumPy (for data manipulation)

Matplotlib, Seaborn (for visualization)

Statsmodels (for ARIMA, decomposition, ADF test)

Scikit-learn (for model evaluation - RMSE)

Joblib (for saving/loading datasets)

🛠️ Project Workflow:

Data Loading: Loaded cleaned dataset using joblib.

Data Visualization: Visualized time trends and closing price history.

Time Series Decomposition: Broke data into trend, seasonal, and residual components.

Stationarity Testing: Used Augmented Dickey-Fuller Test to validate stationarity.

Model Training: Built and forecasted using ARIMA and Simple Exponential Smoothing.

Forecast Visualization: Displayed future stock price predictions with historical trends.

Model Evaluation: Calculated RMSE between actual and predicted values.

Deployment: Deployed using Streamlit with an intuitive interface.

📈 Model Performance

The ARIMA model delivered reasonable forecasts with smooth trend estimation and acceptable RMSE on test data. Users can dynamically choose forecast range and test size within the app.

🧠 Key Features:

View raw data and descriptive statistics

Decompose time series into trend, seasonal, and residual components

Stationarity check using ADF Test

Forecast Netflix stock prices for up to 365 days

Choose between ARIMA and Simple Exponential Smoothing

Interactive Streamlit interface for user inputs and charts



















![Screenshot 2025-05-26 121240](https://github.com/user-attachments/assets/acf62242-c043-44c2-bd92-a58eae6c71e6)

![Screenshot 2025-05-26 121256](https://github.com/user-attachments/assets/877335d6-b8e7-4f39-a7e3-44ef926b92f7)

![Screenshot 2025-05-26 121322](https://github.com/user-attachments/assets/6de886e2-69ae-4e63-b9c8-b5770dab9bf8)

![Screenshot 2025-05-26 121343](https://github.com/user-attachments/assets/f28b5499-2a49-4fd0-839b-6f5f15e8e77f)

![Screenshot 2025-05-26 121453](https://github.com/user-attachments/assets/df565d1f-5d71-46c7-a440-2ab3831614cd)

![Screenshot 2025-05-26 121513](https://github.com/user-attachments/assets/b63a5021-2639-4e59-b344-e56643da6c50)

![Screenshot 2025-05-26 121531](https://github.com/user-attachments/assets/d3b0a4f4-2210-4aab-aa00-33cac2bae8ea)
