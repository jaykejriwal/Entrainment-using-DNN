#Program to use KL divergence as cost function with 10 random variables
import os
import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import transforms
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.utils import shuffle
import h5py
import random
import pdb
import csv
import argparse
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time

st = time.time()
featDim = 512

class EntDataset(Dataset):
    def __init__(self, file_name):
        self.file_name = file_name
        
    def __getitem__(self, idx):
        hf = h5py.File(self.file_name, 'r')
        X = hf['audiodataset'][idx,0:featDim]
        Y = hf['audiodataset'][idx,featDim:2*featDim]
        Z = hf['audiodataset'][idx,0:2*featDim]
        hf.close()
        return (X,Y,Z)

    def __len__(self):
        hf = h5py.File(self.file_name, 'r')
        length=hf['audiodataset'].shape[0]
        hf.close()
        return length

class VAE(nn.Module):
  '''
    Simple Convolutional Neural Network
  '''
  def __init__(self):
        super(VAE, self).__init__()
        self.input_size=512
        self.hidden_size=128
        zdim=30

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, zdim)
        self.bn2 = nn.BatchNorm1d(zdim)

        self.fc3 = nn.Linear(zdim, self.hidden_size)
        self.bn3 = nn.BatchNorm1d(self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.input_size)
        self.bn4 = nn.BatchNorm1d(self.input_size)

        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

  def encode(self, x):
      h1 = self.fc1(x)
      h1= self.relu(self.bn1(h1))
      h2 = self.fc2(h1)
      z = self.bn2(h2)
      return z

  def decode(self, z):
      h3 = self.fc3(z)
      h3 = self.bn3(h3)
      h3 = self.relu(h3)
      h4 = self.fc4(h3)
      h4 = self.bn4(h4)
      return h4

  def forward(self, x):
      z = self.encode(x.view(-1, self.input_size))
      return self.decode(z)

  def embedding(self, x):
      z = self.encode(x.view(-1, self.input_size))
      return z
  
def loss_function(recon_x1, x2):
    input_size = 512
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    BCE = kl_loss(recon_x1, x2.view(-1, input_size))
    return BCE


def lp_distance(x1, x2):
    cos = torch.nn.CosineSimilarity()
    dist = cos(x1, x2)
    return dist

def transfer_dataloader(dataloader, index=0):
    cache_list = list(iter(dataloader))
    assert len(cache_list) > 0
    assert index < len(cache_list[0])
    result_list = np.array(list(map(lambda x: x[index].numpy(), cache_list)))
    return result_list

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def shuffler(X):
    even=X[::2]
    odd=X[1::2]
    random.shuffle(even)
    random.shuffle(odd) 
    shuffled_list=[]
    for e, o in zip(even, odd):
        shuffled_list.append(e)
        shuffled_list.append(o)
    return shuffled_list
  
if __name__ == '__main__':
  
  # Configuration options
  k_folds = 10
  num_epochs = 10
  
  # For fold results
  results = {}
  
  # Set fixed random number seed
  torch.manual_seed(42)
  
  # Prepare MNIST dataset by concatenating Train/Test part; we split later.
  dataset_train_part = EntDataset('/home/jay_kejriwal/Fisher/Processed/h5/CGC/Audio/train_nonorm.h5')
  dataset_val_part = EntDataset('/home/jay_kejriwal/Fisher/Processed/h5/CGC/Audio/val_nonorm.h5')
  dataset_test_part = EntDataset('/home/jay_kejriwal/Fisher/Processed/h5/CGC/Audio/test_nonorm.h5')
  dataset = ConcatDataset([dataset_train_part, dataset_val_part, dataset_test_part])
  
  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True, random_state=27)
    
  # Start print
  print('--------------------------------')

  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=128, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=128, sampler=test_subsampler)
    X_Y_test = next(iter(testloader))[2].numpy()
    Y_test_r = next(iter(testloader))[1].numpy()
    Y_test_rand1=shuffler(Y_test_r)
    Y_test_rand2=shuffler(Y_test_r)
    Y_test_rand3=shuffler(Y_test_r)
    Y_test_rand4=shuffler(Y_test_r)
    Y_test_rand5=shuffler(Y_test_r)
    Y_test_rand6=shuffler(Y_test_r)
    Y_test_rand7=shuffler(Y_test_r)
    Y_test_rand8=shuffler(Y_test_r)
    Y_test_rand9=shuffler(Y_test_r)
    Y_test_rand10=shuffler(Y_test_r)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    model = VAE().double()
    model.apply(weight_reset)
    model.to(device)
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    
    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):

      # Print epoch
      print(f'Starting epoch {epoch+1}')

      # Set train loss value
      train_loss = 0

      # Iterate over the DataLoader for training data
      for i, (data, y_data,z_data) in enumerate(trainloader):
        data = Variable(data)
        y_data = Variable(y_data)
        data = data.to(device)
        y_data = y_data.to(device)  
        # Zero the gradients
        optimizer.zero_grad()
        
        recon_batch = model(data)
        loss = loss_function(recon_batch, y_data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # Print statistics
      train_loss /=  len(trainloader.dataset)    
      print('====> Average loss: {:.4f}'.format(train_loss))

    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')
    
    # Saving the model
    model_pth = f'/home/jay_kejriwal/Fisher/Processed/model/trained_CGC_TRILL_10random{fold}.pt'
    torch.save(model, model_pth)
    # Evaluation for this fold
    model = torch.load(model_pth)
    model.to(device)
    model.eval()
    test_loss = 0
    loss_real = 0
    fake_test_loss = 0
    Loss=[]
    Fake_loss = []
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for idx, (data,Y_test_r1,Y_test_r2,Y_test_r3,Y_test_r4,Y_test_r5,Y_test_r6,Y_test_r7,Y_test_r8,Y_test_r9,Y_test_r10) in enumerate(zip(X_Y_test,Y_test_rand1,Y_test_rand2,Y_test_rand3,Y_test_rand4,Y_test_rand5,Y_test_rand6,Y_test_rand7,Y_test_rand8,Y_test_rand9,Y_test_rand10)):
            x_data = data[:featDim]
            y_data = data[featDim:2*featDim]
            y_fake_data1 = Y_test_r1[:featDim]
            y_fake_data2 = Y_test_r2[:featDim]
            y_fake_data3 = Y_test_r3[:featDim]
            y_fake_data4 = Y_test_r4[:featDim]
            y_fake_data5 = Y_test_r5[:featDim]
            y_fake_data6 = Y_test_r6[:featDim]
            y_fake_data7 = Y_test_r7[:featDim]
            y_fake_data8 = Y_test_r8[:featDim]
            y_fake_data9 = Y_test_r9[:featDim]
            y_fake_data10 = Y_test_r10[:featDim]

            x_data = Variable(torch.from_numpy(x_data))
            y_data = Variable(torch.from_numpy(y_data))
            y_fake_data1 = Variable(torch.from_numpy(y_fake_data1))
            y_fake_data2 = Variable(torch.from_numpy(y_fake_data2))
            y_fake_data3 = Variable(torch.from_numpy(y_fake_data3))
            y_fake_data4 = Variable(torch.from_numpy(y_fake_data4))
            y_fake_data5 = Variable(torch.from_numpy(y_fake_data5))
            y_fake_data6 = Variable(torch.from_numpy(y_fake_data6))
            y_fake_data7 = Variable(torch.from_numpy(y_fake_data7))
            y_fake_data8 = Variable(torch.from_numpy(y_fake_data8))
            y_fake_data9 = Variable(torch.from_numpy(y_fake_data9))
            y_fake_data10 = Variable(torch.from_numpy(y_fake_data10))

            x_data = x_data.to(device)
            y_data = y_data.to(device)
            y_fake_data1 = y_fake_data1.to(device)
            y_fake_data2 = y_fake_data2.to(device)
            y_fake_data3 = y_fake_data3.to(device)
            y_fake_data4 = y_fake_data4.to(device)
            y_fake_data5 = y_fake_data5.to(device)
            y_fake_data6 = y_fake_data6.to(device)
            y_fake_data7 = y_fake_data7.to(device)
            y_fake_data8 = y_fake_data8.to(device)
            y_fake_data9 = y_fake_data9.to(device)
            y_fake_data10 = y_fake_data10.to(device)


            recon_batch = model(x_data)
            z_x = model.embedding(x_data)
            z_y = model.embedding(y_data)
            z_y_fake1 = model.embedding(y_fake_data1)
            z_y_fake2 = model.embedding(y_fake_data2)
            z_y_fake3 = model.embedding(y_fake_data3)
            z_y_fake4 = model.embedding(y_fake_data4)
            z_y_fake5 = model.embedding(y_fake_data5)
            z_y_fake6 = model.embedding(y_fake_data6)
            z_y_fake7 = model.embedding(y_fake_data7)
            z_y_fake8 = model.embedding(y_fake_data8)
            z_y_fake9 = model.embedding(y_fake_data9)
            z_y_fake10 = model.embedding(y_fake_data10)


            loss_real = lp_distance(z_x, z_y).item()
            loss_fake1 = lp_distance(z_x, z_y_fake1).item()
            loss_fake2 = lp_distance(z_x, z_y_fake2).item()
            loss_fake3 = lp_distance(z_x, z_y_fake3).item()
            loss_fake4 = lp_distance(z_x, z_y_fake4).item()
            loss_fake5 = lp_distance(z_x, z_y_fake5).item()
            loss_fake6 = lp_distance(z_x, z_y_fake6).item()
            loss_fake7 = lp_distance(z_x, z_y_fake7).item()
            loss_fake8 = lp_distance(z_x, z_y_fake8).item()
            loss_fake9 = lp_distance(z_x, z_y_fake9).item()
            loss_fake10 = lp_distance(z_x, z_y_fake10).item()

            loss_fake = np.mean([loss_fake1,loss_fake2,loss_fake3,loss_fake4,loss_fake5,loss_fake6,loss_fake7,loss_fake8,loss_fake9,loss_fake10])


            test_loss += loss_real
            fake_test_loss += loss_fake 

            Loss.append(loss_real)
            Fake_loss.append(loss_fake)
        test_loss /= X_Y_test.shape[0]
        fake_test_loss /= X_Y_test.shape[0]
        Loss=np.array(Loss)
        Fake_loss=np.array(Fake_loss)
        test_accuracy=float(np.sum(Loss > Fake_loss))/Loss.shape[0]
        print ("Total Real Loss:"+str(test_loss) + "Total Fake Loss:" + str(fake_test_loss))
        print("Model Accuracy: "+str(test_accuracy*100.0))
        results[fold] = 100.0 * test_accuracy

    
  # Print fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  sum = 0.0
  for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
  print(f'Average: {sum/len(results.items())} %')
  elapsed_time = time.time() - st
  print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
