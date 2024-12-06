# Each code block is separated by #---------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Analysis Libraries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms, models

#miscelaneous
from google.colab import files

#-------------------------------------------------------------------------------

#downloading the colorectal cancer dataset that will be used for training
if not os.path.exists("Dataset_1.zip"):
  !gdown 1nkn9BZ95ODrscLkacvx9-O6ydCrtLOXa
  !unzip Dataset_1.zip
else:
  print("The dataset has already been downloaded. Skipping this step.")

#-------------------------------------------------------------------------------

#preparing and loading the images in the dataset
def loadImages(path, batch_size):
  #firstly defining the transformations used on the images
  transformation = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.RandomRotation(20),
      transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
      transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
      transforms.RandomInvert(), #added in attempt 4
      transforms.RandomPerspective(), #added in attempt 4
      transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  #now loading the images
  dataset = datasets.ImageFolder(path, transform=transformation)

  #defining the sizes of the training, validation, and testing sets. we will be
  #using a 60:20:20 ratio leads to overfitting, try 70:15:15 or 80:10:10
  dataset_size = len(dataset)
  train_size = int(0.7*dataset_size)
  validation_size = int(0.15*dataset_size)
  test_size = dataset_size - (train_size + validation_size)

  #now splitting the dataset into the subsets
  train_set, validation_set, test_set = random_split(dataset, [train_size, validation_size, test_size])

  #creating dataloaders
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

  return train_loader, validation_loader, test_loader

#-------------------------------------------------------------------------------

def save_model(model, optimizer, epoch, path='model_checkpoint.tar'):
  checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
  torch.save(checkpoint, path)

def load_model(model, optimizer, path):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']

  return epoch

#-------------------------------------------------------------------------------

def train_model(model, loss_func, optimizer, train_loader, validation_loader, device, num_epochs):
  train_losses = []
  validation_losses = []

  for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    #Save the state of the model every 20 epochs
    if ((epoch % 20) == 0):
      save_model(model, optimizer, epoch)

    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()

      outputs = model(images)
      train_loss = loss_func(outputs, labels)
      train_loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 1.2)
      optimizer.step()

      epoch_loss += train_loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    #validation step
    model.eval()
    val_loss = 0
    with torch .no_grad():
      for images, labels in validation_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        val_loss += loss_func(outputs, labels).item()

    val_loss /= len(validation_loader)
    validation_losses.append(val_loss)

    #saving model if this was the last epoch
    if (epoch == (num_epochs - 1)):
      save_model(model, optimizer, epoch)

  return train_losses, validation_losses

#-------------------------------------------------------------------------------

def test_model(model, test_loader, device):
  model.eval()
  y_pred = []
  y_true = []

  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      dummy_var, pred_labels = torch.max(outputs, 1)
      y_pred.extend(pred_labels.cpu().numpy())
      y_true.extend(labels.cpu().numpy())

  return y_pred, y_true

#-------------------------------------------------------------------------------

  model = models.resnet34()
  model.fc = nn.Linear(model.fc.in_features, 3)

  #checking if cuda is available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  loss_function = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001) #added weight decay to see if it reduces overfitting

  train_loader, validation_loader, test_loader = loadImages("Colorectal Cancer", 64)

#-------------------------------------------------------------------------------

  model_train_loss, model_validation_loss = train_model(model, loss_function, optimizer, train_loader, validation_loader, device, 100)

  plt.plot(model_train_loss, label=f'Training Loss; Batch Size: {64}, Learning Rate: {0.0001}')
  plt.plot(model_validation_loss, label='Validation Loss')
  plt.title(label="Training Loss over 100 Epochs")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  y_pred, y_true = test_model(model, test_loader, device)
  print(f'Model Accuracy: {accuracy_score(y_true, y_pred)}')
  print(f'Classification report:\n {classification_report(y_true, y_pred)}')
  print(f'Confusion matrix:\n {confusion_matrix(y_true, y_pred)}')

  files.download('model_checkpoint.tar')

#-------------------------------------------------------------------------------

#download the trained weights
#!gdown 12LQ-GVdlMLDK-xmZSkTrtfgVCnEV2AQt
#path = 'attempt8.tar'

#alternate trained weights; uncomment for use
!gdown 1VG6yUF1elECW7lkFN1QXV6VUTtIPCebB
path = 'attempt9.tar'

dummy_epoch = load_model(model, optimizer, path)

#-------------------------------------------------------------------------------

def extract_features_for_tsne(model, test_loader, device):

    model.eval()  # Set model to evaluation mode
    features, labels = [], []
    # Remove the classification layer (last layer) to get only feature outputs
    model_features = nn.Sequential(*list(model.children())[:-1])

    with torch.no_grad():
        for images, target_labels in test_loader:
            images = images.to(device)

            # Remove the final layer for feature extraction (assuming model_features is the model without the last layer)
            outputs = model_features(images)  # Replace model with model_features if the last layer was sliced
            features.append(outputs.view(outputs.size(0), -1).cpu().numpy())  # Flatten feature tensor
            labels.extend(target_labels.cpu().numpy())  # Collect labels on CPU for t-SNE

    # Concatenate all features for t-SNE processing
    features = np.concatenate(features, axis=0)

    return features, labels

#-------------------------------------------------------------------------------

features, labels = extract_features_for_tsne(model, test_loader, device)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(features)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar()
plt.title("t-SNE Visualization of CNN Extracted Features")
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.show()