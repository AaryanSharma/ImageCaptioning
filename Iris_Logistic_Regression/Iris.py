#Importing the libraries
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare data

iris=datasets.load_iris()
x,y = iris.data,iris.target

n_samples,n_features = x.shape

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

sc=StandardScaler()
x_train= sc.fit_transform(x_train)
x_test=sc.transform(x_test)

x_train= torch.from_numpy(x_train.astype(np.float32))
x_test= torch.from_numpy(x_test.astype(np.float32))
y_train= torch.from_numpy(y_train.astype(np.float32))
y_test= torch.from_numpy(y_test.astype(np.float32))

# y_train= y_train.view(y_train.shape[0],1)
# y_test= y_test.view(y_test.shape[0],1)


y_train=F.one_hot(y_train.long(),num_classes=3)

# 1) Model

class Model(nn.Module):
    def __init__(self,n_input_features):
        super(Model,self).__init__()
        self.linear = nn.Linear(n_input_features,3)
    
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model=Model(n_features)    

# 2) Loss and optimizer

num_epochs=100
learning_rate= 0.1
criterion= nn.CrossEntropyLoss()    
optimizer= torch.optim.SGD(model.parameters(),lr=learning_rate)

# 3) Training loop

for epoch in range(num_epochs):
    y_pred=model(x_train)
    loss=criterion(y_pred,y_train.type(torch.float32))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    outputs = model(x_test)
    _, predicted = torch.max(outputs.data, 1)

    # Calculate accuracy
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy:.4f}")
