import json
import numpy as np

# Pretend trained coefficients
coef = 2.5
intercept = 1.0

with open("input.json", "r") as f:
    data = json.load(f)

x = np.array(data["x"]) 
y_pred = coef * x + intercept

output = {"predictions": y_pred.tolist()}
with open("output.json", "w") as f:
    json.dump(output, f)

print("Prediction complete! Saved to output.json")
