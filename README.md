# Developing a Neural Network Regression Model
## NAME:SANJITH R
## REG NO:212223230191
## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The model is a:

Feedforward Artificial Neural Network (ANN)
Also called a Multilayer Perceptron (MLP).

It contains:

Input layer

Two hidden layers

Output layer

## Neural Network Model

<img width="979" height="681" alt="image" src="https://github.com/user-attachments/assets/e23e61de-4635-4b85-bc9b-823efe485612" />



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load Dataset
dataset1 = pd.read_csv('/content/Dhata.csv')
print(dataset1.head(10))
X = dataset1[['Input']].values
y = dataset1[['Output']].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)
# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize Model, Loss and Optimizer
ai_dhivya = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_dhivya.parameters(), lr=0.001)

# Training Function
def train_model(ai_dhivya, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_dhivya(X_train), y_train)
        loss.backward()
        optimizer.step()
        ai_dhivya.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

# Train the Model
train_model(ai_dhivya, X_train_tensor, y_train_tensor, criterion, optimizer)

# Test Evaluation
with torch.no_grad():
    test_loss = criterion(ai_dhivya(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")
print("Name:Dhivya Dharshini B")
print("Reg No :212223240031")
# Plot Loss
loss_df = pd.DataFrame(ai_dhivya.history)
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.show()

# Take user input
user_input = float(input("Enter Input Value: "))

# Convert to tensor
X_new = torch.tensor([[user_input]], dtype=torch.float32)

# Scale input
X_new_scaled = torch.tensor(scaler.transform(X_new), dtype=torch.float32)

# Predict
with torch.no_grad():
    prediction = ai_dhivya(X_new_scaled).item()

print(f"Predicted Output: {prediction}")


```
## Dataset Information

<img width="214" height="252" alt="image" src="https://github.com/user-attachments/assets/9ea7f318-c782-4837-9716-34ce6503e6d1" />


## OUTPUT



### Epoch Loss

<img width="363" height="239" alt="image" src="https://github.com/user-attachments/assets/df277346-74e0-4965-a155-9600bde953c4" />

### Training Loss Vs Iteration Plot


<img width="762" height="620" alt="image" src="https://github.com/user-attachments/assets/15677c83-4e09-41b0-8d66-36487e7371e6" />



### New Sample Data Prediction


<img width="365" height="44" alt="image" src="https://github.com/user-attachments/assets/39944c16-093d-4479-b048-f8a24a5c0fc0" />


## RESULT
Thus a Neural Network Regression Model is developed sucessfully.

