import torch
import torch.nn as nn
import torch.optim as optim

class myModel(nn.Mdodule):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten
        self.seq=nn.Sequential(
            nn.Linear(17,0)
            nn.ReLU()
            nn.Linear(10,5)
            nn.ReLu()
            nn.Linear(5,1)
        )
    def forward(self, x):
        x=self.flatten(x)
        return self.seq(x)
myFirstModel=myModel()
loss_fn=nn.MSELoss()
optimizer=optim.SGD(myFirstModel.parameters(), lr=0.01)
epchs=100
for epch in range(1, epchs+1):
    inputs=torch.randn(1,17)
    labels=torch.randn(1,1)
    optimizer.zero_grad()
    outputs=myFirstModel(inputs)
    loss=loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"epch: {epch}, loss: {loss.item():.4f}")