import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('atlanta_housing_data_clean.csv')

features = ['SizeRank', '2024-10-31', '2024-12-31']
target = '2025-09-30'

X = df[features].values  
y = df[target].values 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SNormalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) 
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define a simple linear regression model using PyTorch
class HousePricePredictor(nn.Module):
    def __init__(self, input_size):
        super(HousePricePredictor, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One output (house price)

    def forward(self, x):
        return self.linear(x)

# Initialize the model
model = HousePricePredictor(input_size=X_train.shape[1])

# Define the (Mean Squared Error) and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
epochs = 1000
loss_values = []

for epoch in range(epochs):
    model.train()  # Set model to training mode

    # FCompute predicted values
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

# Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation for evaluation
    y_test_pred = model(X_test_tensor)

# Convert predictions and actual values to NumPy arrays for evaluation
y_test_pred_np = y_test_pred.numpy()
y_test_np = y_test_tensor.numpy()

# Calculate evaluation metrics
mse = mean_squared_error(y_test_np, y_test_pred_np)
r2 = r2_score(y_test_np, y_test_pred_np)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r2:.4f}')

# Visualize the actual vs predicted values
plt.scatter(y_test_np, y_test_pred_np)
plt.xlabel('Actual House Prices (2025-09-30)')
plt.ylabel('Predicted House Prices (2025-09-30)')
plt.title('Actual vs Predicted House Prices')
plt.show()

# Plot the training loss
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()
