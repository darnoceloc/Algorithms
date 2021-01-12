# Image Classifier experimentation
           
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
from bisect import bisect
import os, psutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
# """ ========================  Functions ============================= """

# functions to show an image
def imshow(img):
    # img = img / + 255
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg*255).astype(np.uint8))
    plt.show()
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
class TensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transforms=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transforms

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = transforms.ToPILImage()(x)
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

# Load image and labels for train and test
class BigDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, target_paths):
        self.data_memmaps = np.load(data_paths, mmap_mode='r+')
        self.target_memmaps = np.load(target_paths, mmap_mode='r+')
        self.start_indices = [0]
        self.data_count = 0
        for index, memmap in enumerate(self.data_memmaps):
            self.start_indices = self.data_count
            self.data_count += memmap.shape[0]

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = self.start_indices
        index_in_memmap = index - self.start_indices
        data = self.data_memmaps
        target = self.target_memmaps
        return index, torch.from_numpy(data), torch.from_numpy(target)

data_paths = '/home/darnoc/Downloads/Images_32.npy'
target_paths = '/home/darnoc/Downloads/Labels_32.npy'

def normalize(x, m, s):
    return (x-m)/s
def normalize_to(data):
    m1, s1 = torch.Tensor.float(data).mean(), torch.Tensor.float(data).std()
    # m2, s2 = torch.Tensor.float(valid).mean(), torch.Tensor.float(valid).std()
    return m1, s1

batch_size = 250
# """ ========================  Entry Point ============================= """
if __name__ == "__main__":

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    dataset = BigDataset(data_paths, target_paths)
    used_memory = process.memory_info().rss - memory_before
    print("Used memory:", used_memory, "bytes")

    dataset_size = len(dataset)
    print("Dataset size:", dataset_size)
    # print("Samples:")
    # for sample_index in [0, dataset_size // 2, dataset_size - 1]:
    #     print(dataset[sample_index])

    train_images = dataset.data_memmaps
    train_labels = dataset.target_memmaps
    # Fix the RGB values
    train_images_fixed = []
    train_labels_fixed = []
    for i in range(train_images.size):
        train_labels_fixed.append(train_labels[i])
        image = train_images[i]
        train_images_fixed.append((image*-255).astype(np.uint8))
    train_images_fixed = np.stack(train_images_fixed)
    train_labels_fixed = np.array(train_labels_fixed)

    train_x, val_x, train_y, val_y = train_test_split(train_images_fixed, train_labels_fixed, test_size=0.2)
    print("Train image shape", train_x.shape)
    print("Test image shape", val_x.shape)

    train_tensor_x = torch.from_numpy(np.resize(train_x, (train_y.size, 1, 200, 200))) # transform to torch tensor
    train_tensor_y = torch.from_numpy(train_y.astype(int))

    test_tensor_x = torch.from_numpy(np.resize(val_x, (val_y.size, 1, 200, 200)))  # transform to torch tensor
    test_tensor_y = torch.from_numpy(val_y.astype(int))

    x_train_mean, x_train_std = normalize_to(train_tensor_x)
    trans1 = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((x_train_mean,), (x_train_std,))])
    trainset = TensorDataset(tensors=(train_tensor_x, train_tensor_y), transforms=trans1)  # create your datset
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)  # create your dataloader

    x_test_mean, x_test_std = normalize_to(test_tensor_x)
    trans2 = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((x_test_mean,), (x_test_std,))])
    testset = TensorDataset(tensors=(test_tensor_x, test_tensor_y), transforms=trans2)  # create your datset
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=False)  # create your dataloader

    classes = np.array(("other", "flemish", "english", "stretcher", "otherBrick"))

    # get some random training images
    images, labels = next(iter(trainloader))
    # #show images
    imshow(torchvision.utils.make_grid(images))
    print("Images", images)
    # #print labels
    print("Labels", labels)
    labels = labels.long()
    print(' '.join('%5s' % classes[labels[j]] for j in range(5)))

    net = Net()
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('Started Training')

    for epoch in range(25):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i == len(trainloader):    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i, running_loss / len(trainloader.dataset)))
                running_loss = 0.0

    print('Finished Training')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the ', len(testloader.dataset), ' test images: %d %%' % (
            100 * correct / total))

    images, labels = next(iter(trainloader))
    output_train = net(images)
    softmax = torch.exp(output_train).cpu()
    prob = list(softmax.detach().numpy())
    predictions = np.argmax(prob, axis=1)

    # accuracy on training set
    print("training score", accuracy_score(labels, predictions))

    images, labels = next(iter(testloader))
    output_val = net(images)
    softmax = torch.exp(output_val).cpu()
    prob = list(softmax.detach().numpy())
    predictions = np.argmax(prob, axis=1)

    # accuracy on test set
    print("test score", accuracy_score(labels, predictions))


# Defines the number of times the NN should look at the entire loop
epochs = 25

for epch in range(epochs):
    total_loss = 0
    correct_predictions = 0
    for batch in trainloader:
        # Get the images and labels in a batchwise manner
        images, labels = batch

        # Forward pass
        predictions = net(images)

        # Loss calculation
        loss = F.cross_entropy(predictions, labels.long())

        # Accumulate the loss
        total_loss += loss.item()

        # Zero out the gradients.If you don't do this, gradients will get accumulated and
        # the updates will happen incorrectly.
        optimizer.zero_grad()

        # Backpropagate the loss
        loss.backward()

        # Update the weights of the model
        optimizer.step()

        # Get number of correct predictions on the train set
        predictions = predictions.argmax(dim=1)
        batch_correct = predictions.eq(labels).sum().item()
        correct_predictions += batch_correct

    print(f"Epoch: {epch+1}, Total Loss: {np.round(total_loss, 2)}, Correct Predictions: {correct_predictions}",
          f"Accuracy: {correct_predictions/len(trainloader.dataset)}")

# def get_predictions(model=None, loader=None):
#     # Define an empty container
#     all_preds = torch.tensor([], dtype=torch.int64)
#
#     # Since we're not training, we turn off the gradient computation graph
#     # This saves time and space both
#     torch.set_grad_enabled(False)
#
#     for batch in loader:
#         # Get a batch of images
#         images, labels = batch
#
#         # Predictions from the images (probabilities)
#         preds = model(images)
#
#         # Get the class labels from predictions
#         predictions = preds.argmax(dim=1)
#
#         # Combine the batch's predictions into a single tensor all_preds
#         all_preds = torch.cat((all_preds, predictions))
#
#     # Set the gradient computation back on which you'd turned off earlier
#     torch.set_grad_enabled(True)
#
#     return all_preds
#
# print(get_predictions(net, testloader))

