# Click-Through-Rate-Prediction

This Streamlit app predicts whether a user will click on an ad based on several features provided by the user. The app uses multiple machine learning models, including Logistic Regression, Random Forest, XGBoost, and LightGBM, to make predictions.

Features:
- User Input: Users can input the following features through the sidebar:
  - Daily Time Spent on Site (minutes
  - Age
  - Area Income ($)
  - Daily Internet Usage (minutes)
  
- Model Selection: Users can select one of four machine learning models (Logistic Regression, Random Forest, XGBoost, or LightGBM) for the prediction.

- Prediction Output: Based on the selected model, the app will predict whether the user is likely to click on the ad or not.

Requirements:-

To run this app locally, you need to install the required dependencies. You can install them by running the following command:

```
pip install -r requirements.txt
```

Here is a list of the required libraries:
- Streamlit For creating the web interface.
- Pandas For handling and manipulating input data.
- Joblib For loading the pre-trained models.
- Scikit-learn For logistic regression and scaling.
- XGBoost For the XGBoost model.
- LightGBM: For the LightGBM model.

Project Structure:-

```
CTR_Project/
│
├── app.py                # The main Streamlit application file
├── models/               # Folder containing saved machine learning models
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   └── scaler.pkl
├── requirements.txt      # List of dependencies for the project
└── README.md             # This file
```

How to Run the App Locally:-

1. Clone the repository:
   ```
   git clone https://github.com/your-username/CTR_Project.git
   cd CTR_Project
   ```

2. Install dependencies
   Run the following command to install all necessary libraries:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   Start the app by running:
   ```bash
   streamlit run app.py
   ```

4. View the app:
   After running the above command, open your browser and navigate to `http://localhost:8501` to interact with the app.

Model Details

The app uses four machine learning models for predicting ad clicks:
1. Logistic Regression: A simple linear model for binary classification.
2. Random Forest :An ensemble model using multiple decision trees.
3. XGBoost: A gradient boosting algorithm.
4. LightGBM: A gradient boosting framework optimized for efficiency.

These models are pre-trained and stored in the `models/` folder.

Notes:
- File Paths Ensure that the file paths for loading models and scaler in `app.py` are correct for your system. Update the file paths if needed.
- Model Files: The `models/` folder contains the pre-trained models and scaler needed for predictions. Ensure these files are present and correctly named in your repository.

Contributing

If you would like to contribute to this project, feel free to open an issue or submit a pull request with improvements.

License:-

This project is open-source and available under the MIT License.

