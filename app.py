import streamlit as st
import pandas as pd
import joblib

try:
    logistic_model = joblib.load('C:\\Users\\HP\\OneDrive\\Desktop\\Python Folder\\CTR\\models\\logistic_model.pkl')
    rf_model = joblib.load('C:\\Users\\HP\\OneDrive\\Desktop\\Python Folder\\CTR\\models\\random_forest_model.pkl')
    scaler = joblib.load('C:\\Users\\HP\\OneDrive\\Desktop\\Python Folder\\CTR\\models\\scaler.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def main():
    st.title("📊 Ad Click Prediction")
    st.markdown(
        """
        This app predicts whether a user is likely to **click on an ad** based on their data.
        Use the sidebar to input features, select a model, and click **Predict**!
        """
    )
    
    st.sidebar.header("User Input Features")
    daily_time = st.sidebar.number_input("🕒 Daily Time Spent on Site (minutes)", min_value=0.0, value=50.0)
    age = st.sidebar.number_input("👤 Age", min_value=1, value=30)
    area_income = st.sidebar.number_input("💵 Area Income ($)", min_value=0.0, value=60000.0)
    daily_internet = st.sidebar.number_input("🌐 Daily Internet Usage (minutes)", min_value=0.0, value=200.0)
    model_choice = st.sidebar.selectbox("🤖 Choose Model", ["Logistic Regression", "Random Forest"])

    
    income_per_age = area_income / age if age > 0 else 0
    time_per_internet = daily_time / daily_internet if daily_internet > 0 else 0

    
    input_data = pd.DataFrame(
        [[daily_time, age, area_income, daily_internet, income_per_age, time_per_internet]],
        columns=['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Income_Per_Age', 'Time_Per_Internet']
    )
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### User Input Summary")
        st.write(input_data)

    with col2:
        st.markdown("### Prediction")
        if st.button("Predict"):
            # Scale input data
            try:
                input_data_scaled = scaler.transform(input_data)
            except Exception as e:
                st.error(f"Error scaling input data: {e}")
                st.stop()

            # Make predictions
            try:
                if model_choice == "Logistic Regression":
                    probabilities = logistic_model.predict_proba(input_data_scaled)
                else:  # Random Forest
                    probabilities = rf_model.predict_proba(input_data_scaled)

                prediction = "Clicked on Ad" if probabilities[0][1] > 0.5 else "Did not Click on Ad"
                st.write(f"**Prediction:** {prediction}")
                st.write(f"**Confidence:** {probabilities[0][1]*100:.2f}% for 'Clicked on Ad'")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
