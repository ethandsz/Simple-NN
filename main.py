import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 100, bias=True)
        self.fc2 = nn.Linear(100, 1, bias=True)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
def generate_data(n=100):
    x = torch.linspace(-10, 10, n).view(-1, 1)  # Input tensor (reshaped to column vector)
    y = torch.sin(x)  # Target is the f(x) = sin(x)
    return x, y

        
my_nn = Net()

learning_rate = 0.1
epochs = 1000

# Initialize the neural network, loss function, and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(my_nn.parameters(), lr=learning_rate)

# Visualize the model fitting at every 100 epochs
save = True

# Generate training data
x_train, y_train = generate_data()

# Training loop
losses =[]
for epoch in range(epochs):
    my_nn.train()
    
    y_pred = my_nn(x_train)
    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        if(save):
            my_nn.eval()  # Put the model in evaluation mode
            with torch.no_grad():  # No need to calculate gradients during evaluation
                y_pred = my_nn(x_train)
            plt.plot(x_train, y_train, label='Actual function', color='blue')
            plt.legend(loc='lower left')
            plt.twinx()
            plt.twiny()
            plt.plot(x_train, y_pred, label='NN prediction', color='red')
            plt.legend(loc='lower right')
            plt.title(f'Neural Network Fitting f(x) = sin(x)\nEpoch = {epoch}')
            plt.show()


my_nn.eval()  # Put the model in evaluation mode
with torch.no_grad():  # No need to calculate gradients during evaluation
    y_pred = my_nn(x_train)
plt.plot(x_train, y_train, label='Actual function', color='blue')
plt.plot(x_train, y_pred, label='NN prediction', color='red')
plt.legend()
plt.title(f'Neural Network Fitting f(x) = sin(x)')
plt.show()
