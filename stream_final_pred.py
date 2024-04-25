import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

# Function to filter data by state
def filter_by_state(data, state_name):
  return data[data["SUBDIVISION"] == state_name]

# Function to predict rainfall
def predict_rainfall(state_data, input_year):
  features = ['YEAR']
  target = 'ANNUAL'
  X_train, X_test, y_train, y_test = train_test_split(state_data[features], state_data[target], test_size=0.2, random_state=42)
  model = LinearRegression()
  model.fit(X_train, y_train)
  predicted_rainfall = model.predict([[input_year]])[0]
  return predicted_rainfall

# Function to predict temperature
def predict_temperature(data, input_year):
  features = ['YEAR']
  target = 'ANNUAL'
  X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
  model = XGBRegressor()
  model.fit(X_train, y_train)
  predicted_temperature = model.predict([[input_year]])[0]
  return predicted_temperature

# Function to get groundwater level
def get_groundwater_level(data, state, district):
  filtered_data = data[(data['Name of State'] == state) & (data['Name of District'] == district)]
  if filtered_data.empty:
    return None
  else:
    groundwater_level = filtered_data['Net Ground Water Availability for future use'].values[0]
    return groundwater_level

# Function to determine drought level
def determine_drought_level(predicted_rainfall, predicted_temperature, groundwater_level, rainfall_threshold=0, temperature_threshold=0, groundwater_threshold=0):
  # Normalize predicted temperature
  scaler = MinMaxScaler(feature_range=(0, 1))
  normalized_temperature = scaler.fit_transform([[predicted_temperature]])[0][0]
  # Assign scores based on thresholds
  rainfall_score = 1 if predicted_rainfall < rainfall_threshold else 0
  temperature_score = 1 if normalized_temperature > 0.5 else 0  # Adjust threshold based on normalization
  groundwater_score = 1 if groundwater_level < groundwater_threshold else 0
  # Combine scores
  overall_score = rainfall_score + temperature_score + groundwater_score
  # Determine drought level based on overall score
  if overall_score == 0:
    drought_level = "No Drought"
  elif overall_score == 1:
    drought_level = "Mild Drought"
  elif overall_score == 2:
    drought_level = "Moderate Drought"
  else:
    drought_level = "Severe Drought"
  return drought_level

# Load data
data_rainfall = pd.read_csv("C:/Users/ADMIN/Downloads/rainfall1.csv")
data_temperature = pd.read_csv("C:/Users/ADMIN/Downloads/temperature_data.csv")
data_groundwater = pd.read_csv("C:/Users/ADMIN/Downloads/grnd_water_level.csv")

# Streamlit App with Background Image
def main():
  # Set background image using CSS
    background_image = """
  <style>
  [data-testid="stAppViewContainer"] {
      background-image: url("https://e1.pxfuel.com/desktop-wallpaper/352/153/desktop-wallpaper-nice-backgrounds-extra-nice-background-pic-thumbnail.jpg");
      background-size: cover;
  }
  </style>
  """
    st.markdown(background_image, unsafe_allow_html=True)

  # Set page title
    st.title('Drought Prediction')

  # Get user input for state and district
    state_input = st.text_input('Enter the name of the state (region):')
    district_input = st.text_input('Enter the name of the district:')
    input_year = st.number_input('Enter the year:', min_value=1901, max_value=2050, value=2022, step=1)

    if st.button('Predict Drought Level'):
        # Filter data by state
        state_data_rainfall = filter_by_state(data_rainfall, state_input)
        if state_data_rainfall.empty:
            st.write("No data found for the specified state.")
        else:
            # Predict rainfall
            predicted_rainfall = predict_rainfall(state_data_rainfall, input_year)
            st.write(f"Predicted Annual Rainfall for {state_input} ({input_year}): {predicted_rainfall} mm")

            # Predict temperature
            predicted_temperature = predict_temperature(data_temperature, input_year)
            st.write(f"Predicted Temperature for Year {input_year}: {predicted_temperature} °C")

            # Get groundwater level
            groundwater_level = get_groundwater_level(data_groundwater, state_input, district_input)

            # Determine drought level
            drought_level = determine_drought_level(predicted_rainfall, predicted_temperature, groundwater_level)
            st.write(f"Drought Level for {state_input}: {drought_level}")

if __name__ == '__main__':
    main()