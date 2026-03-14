import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load the data
data = pd.read_csv("data/insurance.csv")

# Display basic info
print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())
print(data.head())

# Preprocess categorical variables
categorical_cols = ["sex", "smoker", "region"]
encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_features = encoder.fit_transform(data[categorical_cols])

# Create feature matrix
X = pd.concat(
    [
        data[["age", "bmi", "children"]],
        pd.DataFrame(
            encoded_features, columns=encoder.get_feature_names_out(categorical_cols)
        ),
    ],
    axis=1,
)

y = data["charges"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Save the model
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Model saved as model.pkl and encoder.pkl")


# Function to make predictions
def predict_charges(age, sex, bmi, children, smoker, region):
    """
    Predict insurance charges based on input features.

    Parameters:
    age (int): Age of the person
    sex (str): 'male' or 'female'
    bmi (float): Body mass index
    children (int): Number of children
    smoker (str): 'yes' or 'no'
    region (str): 'northeast', 'northwest', 'southeast', 'southwest'

    Returns:
    float: Predicted charges
    """
    # Load model and encoder
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")

    # Prepare input data
    input_data = pd.DataFrame(
        {
            "age": [age],
            "bmi": [bmi],
            "children": [children],
            "sex": [sex],
            "smoker": [smoker],
            "region": [region],
        }
    )

    # Encode categorical variables
    encoded_input = encoder.transform(input_data[["sex", "smoker", "region"]])

    # Create feature matrix
    X_input = pd.concat(
        [
            input_data[["age", "bmi", "children"]],
            pd.DataFrame(
                encoded_input, columns=encoder.get_feature_names_out(categorical_cols)
            ),
        ],
        axis=1,
    )

    # Make prediction
    prediction = model.predict(X_input)[0]
    return prediction


# Example usage
if __name__ == "__main__":
    example_prediction = predict_charges(30, "male", 25.0, 2, "no", "northeast")
    print(f"Example prediction: ${example_prediction:.2f}")
