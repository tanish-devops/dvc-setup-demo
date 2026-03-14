  # Insurance Charges Prediction Model

This project contains a basic machine learning model to predict insurance charges based on personal attributes.

## Data

The model uses the insurance.csv dataset with the following features:
- age: Age of the person
- sex: Gender (male/female)
- bmi: Body Mass Index
- children: Number of children
- smoker: Smoking status (yes/no)
- region: Geographic region (northeast, northwest, southeast, southwest)
- charges: Insurance charges (target variable)

## Model

- Algorithm: Random Forest Regressor
- Performance: R-squared ≈ 0.86 on test set
- Features: All input features with categorical variables one-hot encoded

## Usage

### Training
Run the training script:
```bash
python train.py
```

This will:
- Load and preprocess the data
- Train the model
- Evaluate performance
- Save the model and encoder

### Making Predictions

Use the `predict_charges` function in your code:

```python
from train import predict_charges

# Example prediction
charges = predict_charges(
    age=30,
    sex="male",
    bmi=25.0,
    children=2,
    smoker="no",
    region="northeast"
)
print(f"Predicted charges: ${charges:.2f}")
```

## Files

- `train.py`: Training script and prediction function
- `model.pkl`: Trained Random Forest model
- `encoder.pkl`: One-hot encoder for categorical variables
- `data/insurance.csv`: Training data
- `requirements.txt`: Python dependencies
