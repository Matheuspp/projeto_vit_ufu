import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from vit_pytorch import SimpleViT
from torchvision.models import resnet50

from vit_pytorch.distill import DistillableViT, DistillWrapper



def val_prepare(val_size, train_set):
    # preparando conjunto de validação
    data_points = len(train_set)
    indices = list(range(data_points))
    np.random.shuffle(indices)
    split = int(np.floor(val_size * data_points))
    train_idx, valid_idx = indices[split:], indices[:split]
    return SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
    

batch_size = 32

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalize the input data using ImageNet statistics
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Load the datasets
train_dataset = torchvision.datasets.ImageFolder(
    root='fruits-360_dataset/fruits-360/Training',
    transform=transform
)
test_dataset = torchvision.datasets.ImageFolder(
    root='fruits-360_dataset/fruits-360/Test',
    transform=transform
)
tSamp, vSamp = val_prepare(0.2, train_dataset)
# Create data loaders
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=tSamp, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(train_dataset, sampler=vSamp, batch_size=batch_size)

# Initialize the ResNet-18 model
#model = torchvision.models.resnet18(pretrained=False)
num_classes = len(train_dataset.classes)
#model.fc = nn.Linear(model.fc.in_features, num_classes)
#model.to(device)
##======================================================================
teacher = resnet50(pretrained = True)

v = DistillableViT(
    image_size = 100,
    patch_size = 32,
    num_classes = num_classes,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

distiller = DistillWrapper(
    student = v,
    teacher = teacher,
    temperature = 3,           # temperature of distillation
    alpha = 0.5,               # trade between main loss and distillation loss
    hard = False               # whether to use soft or hard distillation
)


model = distiller

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Training loop
num_epochs = 200
min_loss = np.inf
for epoch in range(num_epochs):
    print(f'training... epoch {epoch}')
    running_loss = 0.0
    val_loss = 0.0
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        print(inputs.shape)
        print(labels.shape)

        loss = model(inputs, labels)
        #loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    model.eval()   
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            vloss = model(inputs, labels)
            #vloss = criterion(outputs, labels)
            val_loss += vloss.item()
        
    if val_loss < min_loss:
        min_loss = val_loss
        torch.save(model, 'model_vit.pth')
        print(f'saving model at epoch {epoch}')
        print(f'Epoch {epoch + 1}, Batch {i + 1}: loss {running_loss / 200:.3f} val_loss {val_loss / 200:.3f}')
            

print('Training finished!')
"""
# Evaluation on the test set
model.eval()  # Switch to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
"""
