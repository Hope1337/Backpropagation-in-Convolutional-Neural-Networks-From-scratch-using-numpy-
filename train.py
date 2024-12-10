import torch
from torchvision import datasets, transforms
import sys
import numpy as np

batch_size = 1
epochs     = 5
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

from model import Model, Conv2D
model = Model(input_dim=1, eta=1e-4)

for epoch in range(epochs):
    for idx, (images, labels) in enumerate(train_loader):
        print("epoch : {}, idx : {}".format(epoch, idx))
        images = images.numpy().astype(np.float64)
        labels = labels.numpy().astype(np.float64)

        output = model(images)
        #output2 = model.torch(images)
        #print(output.shape)
        #print(output2.shape)
        #print(Conv2D.norm(torch.from_numpy(output), output2))
        t = np.argmax(output, axis=1)
        t = np.squeeze(t)
        print(t)
        print(labels.astype(int))
        
        loss = model.calc_loss(output, labels)
        print("loss = {}".format(np.sum(loss)))
        model.backward()

        if idx % 500 == 0:
            model.save_weight()
        print()
        