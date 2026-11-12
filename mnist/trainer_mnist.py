import matplotlib.pyplot as plt #for viewing images

from email.mime import image
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#print(torch.__version__);




## 1 -- data loading pipeline / input pipeline 

transformer = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transformer)
test_data = datasets.MNIST(root="data", train = False, download=True, transform = transformer)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle = False)
images,labels = next(iter(train_loader))
#print(images.shape, labels.shape)  # shows the structure of tensors images and labels




## 2  -- for displaying the images --

#plt.imshow(images[4][0], cmap="gray")
#plt.title(f"Label: {labels[4].item()}")
#plt.show()



## 3 -- creating the model -- 

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),

        )

    def forward(self, x):
        return self.layers(x)


model = net()
loss_fn = nn.CrossEntropyLoss() #grader for the output
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #optimizer for updating the weights of the model

model = net()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#images, labels = next(iter(train_loader))
#scores = model(images)
#loss = loss_fn(scores, labels)
#print(loss.item())

#images, labels = next(iter(train_loader))
#print(model(images).shape)


for epoch in range(3):
    for images, labels in train_loader:
        scores = model(images)
        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")





