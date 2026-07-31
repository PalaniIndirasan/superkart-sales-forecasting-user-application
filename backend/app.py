# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_forecast_api = Flask("SuperKart Sales Forecast")

# Load the trained machine learning model
model = joblib.load("superkart_sales_forecast_model_v1_0.joblib")

# Define a route for the home page (GET request)
@superkart_sales_forecast_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Forecast Prediction API!"

# Define an endpoint for single property prediction (POST request)
@superkart_sales_forecast_api.post('/v1/sales')
def predict_rental_price():
    """
    This function handles POST requests to the '/v1/sales' endpoint.
    It expects a JSON payload containing shop details and returns
    the predicted sales price as a JSON response.
    """
    # Get the JSON data from the request body
    sales_data = request.get_json()

    # Extract relevant features and apply preprocessing as done during training
    sample = {
            'Product_Weight': float(sales_data['Product_Weight']),
            'Product_Sugar_Content': sales_data['Product_Sugar_Content'],
            # Apply log1p transformation to Product_Allocated_Area
            'Product_Allocated_Area_Log': np.log1p(float(sales_data['Product_Allocated_Area'])),
            'Product_MRP': float(sales_data['Product_MRP']),
            'Store_Size': sales_data['Store_Size'],
            'Store_Location_City_Type': sales_data['Store_Location_City_Type'],
            'Store_Type': sales_data['Store_Type'],
            'Product_Type_Category': sales_data['Product_Type_Category']
    }

    # Define the expected column order to match X_train used during model training
    expected_columns = [
        'Product_Weight',
        'Product_Sugar_Content',
        'Product_MRP',
        'Store_Size',
        'Store_Location_City_Type',
        'Store_Type',
        'Product_Allocated_Area_Log',
        'Product_Type_Category'
    ]

    # Convert the extracted data into a Pandas DataFrame and enforce column order
    input_data = pd.DataFrame([sample], columns=expected_columns)

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    # Convert predicted_price to Python float
    predicted_price = round(float(predicted_price), 2)

    # Return the actual price
    return jsonify({'Predicted Price (in dollars)': predicted_price})


# Define an endpoint for batch prediction (POST request)
@superkart_sales_forecast_api.post('/v1/salesbatch')
def predict_rental_price_batch():
    """
    This function handles POST requests to the '/v1/salesbatch' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted rental prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_df = pd.read_csv(file)

    # Apply preprocessing steps to match the model's training data
    # 1. Standardize inconsistent category labels for Product_Sugar_Content
    if 'Product_Sugar_Content' in input_df.columns:
        input_df['Product_Sugar_Content'] = input_df['Product_Sugar_Content'].replace({'reg': 'Regular'})

    # 2. Transform Product_Allocated_Area to Product_Allocated_Area_Log
    if 'Product_Allocated_Area' in input_df.columns:
        input_df['Product_Allocated_Area_Log'] = np.log1p(input_df['Product_Allocated_Area'])
        input_df = input_df.drop(columns=['Product_Allocated_Area'])

    # 3. Group Product_Type into Perishables vs. Non Perishables
    if 'Product_Type' in input_df.columns:
        perishable_types = ['Dairy', 'Fruits and Vegetables', 'Meat', 'Breads', 'Seafood']
        input_df['Product_Type_Category'] = input_df['Product_Type'].apply(
            lambda x: 'Perishables' if x in perishable_types else 'Non Perishables'
        )
        input_df = input_df.drop(columns=['Product_Type'])

    # 4. Drop unnecessary columns that were dropped during training
    # The 'id' column is specific to the batch example description, if present in CSV, it should be ignored by model.
    columns_to_drop = ["Product_Id", "Store_Id", "Store_Establishment_Year", "id", "Product_Id_char", "Store_Age_Years"]
    input_df = input_df.drop(columns=columns_to_drop, errors='ignore')

    # Make predictions for all properties in the DataFrame
    predicted_prices_raw = model.predict(input_df).tolist()

    # Round to 2 decimal places
    predicted_prices = [round(float(price), 2) for price in predicted_prices_raw]

    # Return the predictions as a list in a JSON response
    return jsonify({'predictions': predicted_prices})

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_forecast_api.run(debug=True)
