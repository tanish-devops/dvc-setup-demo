from train import predict_charges

# Test predictions
test_cases = [
    (25, "female", 22.5, 0, "no", "southwest"),
    (40, "male", 30.0, 2, "yes", "northeast"),
    (35, "female", 25.0, 1, "no", "northwest"),
]

for age, sex, bmi, children, smoker, region in test_cases:
    prediction = predict_charges(age, sex, bmi, children, smoker, region)
    print(
        f"Age {age}, {sex}, BMI {bmi}, {children} children, smoker: {smoker}, {region}: ${prediction:.2f}"
    )
