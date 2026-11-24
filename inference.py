import torch
from model import LSTMModel

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

checkpoint = torch.load(
    'lstm_model.pt',
    map_location=device,
    weights_only=True
)

model = LSTMModel(
    width=checkpoint['width'],
    depth=checkpoint['depth']
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load test dataset
data = np.load('test_dataset.npz')
X_test = data['X_test']
y_test = data['y_test']

# Convert inputs to PyTorch tensors
X_test_tensor = torch.tensor(
    X_test[:, :, None],
    dtype=torch.float32
).to(device)

# Run predictions
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy()

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Test MSE:  {mse:.5f}")
print(f"Test MAE:  {mae:.5f}")
print(f"Test RMSE: {rmse:.5f}")
print(f"Test R2:   {r2:.5f}")

# Display predictions vs actual
results = np.column_stack((y_test.flatten(), predictions.flatten()))

print("\nActual vs Predicted:")
print("Actual\t\tPredicted")

for actual, predicted in results:
    print(f"{actual:.5f}\t{predicted:.5f}")