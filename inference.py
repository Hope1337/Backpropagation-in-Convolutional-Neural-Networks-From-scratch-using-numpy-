from model import Model
import torch
from torchvision import datasets, transforms
import sys
import numpy as np
import matplotlib.pyplot as plt


batch_size = 1
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

from model import Model, Conv2D
model = Model(input_dim=1, eta=1e-4)
model.load_weight()

for idx, (images, labels) in enumerate(test_loader):
    images = images.numpy().astype(np.float64)
    labels = labels.numpy().astype(np.float64)

    output = model(images)
    print(images.shape)
    
    t = np.argmax(output, axis=1)
    t = np.squeeze(t)
    print("Predict : {}".format(t))
    print("Truth : {}".format(labels.astype(int)))
    plt.imshow(images.reshape(28, 28), cmap='gray')
    plt.axis('off')  # Tắt trục toạ độ
    plt.show()
        
    print()
        

model = Model(1)