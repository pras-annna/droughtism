import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('rainfall_prediction_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Function to predict rainfall
def predict_rainfall(year, state):
    # For demonstration purposes, assuming the state also affects the rainfall prediction
    # You may replace this with your actual logic
    predicted_rainfall = loaded_model.predict([[year]])[0] + len(state)
    return predicted_rainfall

# Streamlit App
def main():
    background_image = """
  <style>
  [data-testid="stAppViewContainer"] {
      background-image: url("https://as1.ftcdn.net/v2/jpg/03/09/05/32/1000_F_309053256_fIE3WK7PGJIATZf0ackF9ZZyTJm9kizF.jpg");
      background-size: cover;
  }
  </style>
  """
    st.markdown(background_image, unsafe_allow_html=True)

    # Set page title
    st.title('Rainfall Prediction')

    # Add sidebar with title and description
    st.sidebar.title('About')
    st.sidebar.info('This app predicts the annual rainfall for a given year and state.')

    # Add input for the year
    year = st.slider('Select a year:', min_value=2000, max_value=2100, value=2022, step=1)

    # Add input for the state
    state = st.text_input('Enter the state:', 'Enter state here')

    # Predict rainfall for the selected year and state
    predicted_rainfall = predict_rainfall(year, state)

    # Display the predicted rainfall
    st.subheader('Prediction Result')
    st.write(f'Year: {year}')
    st.write(f'State: {state}')
    st.write(f'Predicted Annual Rainfall: {predicted_rainfall} mm')

if __name__ == '__main__':
    main()
