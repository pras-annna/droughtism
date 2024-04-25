import streamlit as st
import pickle

# Load the saved model
with open('Temperature_prediction_model.pkl', 'rb') as file:
  loaded_model = pickle.load(file)

# Function to predict temperature
def predict_temperature(year):
  predicted_temperature = loaded_model.predict([[year]])[0]
  return predicted_temperature

# Streamlit App with Background Image
def main():
  # Set background image using CSS
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
  st.title('Temperature Prediction')

  # Add sidebar with title and description
  st.sidebar.title('About')
  st.sidebar.info('This app predicts the annual temperature for a given year.')

  # Add input box for the year
  year = st.text_input('Enter a year (2001-2100):', '2022')

  # Convert input to integer
  year = int(year) - 100

  # Predict temperature for the selected year
  predicted_temperature = predict_temperature(year)

  # Display the predicted temperature
  st.subheader('Prediction Result')
  st.write(f'Year: {year+100}')
  st.write(f'Predicted Annual Temperature: {predicted_temperature} °C')

if __name__ == '__main__':
  main()