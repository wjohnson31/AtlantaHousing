import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the cleaned dataset
df = pd.read_csv('atlanta_housing_data_clean.csv')

# Step 2: Select relevant features (you can modify this based on the features you want to include)
# Here we are using columns like 'SizeRank', 'RegionID', and '2024-10-31' as features
# Target variable is '2025-09-30' (modify this as per your target column)
features = ['SizeRank', '2024-10-31', '2024-12-31']
target = '2025-09-30'

X = df[features].values  # Features
y = df[target].values    # Target: House price predictions for 2025-09-30

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Normalize the data (scaling features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Convert the data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshape target to be a column vector
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Step 6: Define a simple linear regression model using PyTorch
class HousePricePredictor(nn.Module):
    def __init__(self, input_size):
        super(HousePricePredictor, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One output (house price)

    def forward(self, x):
        return self.linear(x)

# Initialize the model
model = HousePricePredictor(input_size=X_train.shape[1])

# Step 7: Define the loss function (Mean Squared Error) and optimizer (Stochastic Gradient Descent)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 8: Train the model
epochs = 1000
loss_values = []

for epoch in range(epochs):
    model.train()  # Set model to training mode

    # Forward pass: Compute predicted values
    y_pred = model(X_train_tensor)
    
    # Compute the loss
    loss = criterion(y_pred, y_train_tensor)
    loss_values.append(loss.item())

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Backpropagate
    optimizer.step()       # Update weights

    # Print loss occasionally
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 9: Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation for evaluation
    y_test_pred = model(X_test_tensor)

# Convert predictions and actual values to NumPy arrays for evaluation
y_test_pred_np = y_test_pred.numpy()
y_test_np = y_test_tensor.numpy()

# Step 10: Calculate evaluation metrics
mse = mean_squared_error(y_test_np, y_test_pred_np)
r2 = r2_score(y_test_np, y_test_pred_np)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r2:.4f}')

# Step 11: Visualize the actual vs predicted values
plt.scatter(y_test_np, y_test_pred_np)
plt.xlabel('Actual House Prices (2025-09-30)')
plt.ylabel('Predicted House Prices (2025-09-30)')
plt.title('Actual vs Predicted House Prices')
plt.show()

# Step 12: Plot the training loss
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()
