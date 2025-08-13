import json
import numpy as np

coef = 2.5
intercept = 1.0

with open("input.json", "r") as f:
    data = json.load(f)

# Ensure we get a list of floats
if isinstance(data["x_values"], str):
    x_values = [float(x) for x in data["x_values"].split(",")]
else:
    x_values = [float(x) for x in data["x_values"]]

x = np.array(x_values)
y_pred = coef * x + intercept

output = {"predictions": y_pred.tolist()}
with open("output.json", "w") as f:
    json.dump(output, f)

print("Prediction complete! Saved to output.json")
