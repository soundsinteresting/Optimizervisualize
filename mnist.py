import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from network import Resnet, Twohiddenlayerfc, CNNMnist
import re
import os

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_some_pictures(j ):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(j)))


def random_test():
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))



def train(net, noe, source_dir, filepath, initial_lr,  batchsize, beta_2, device):
    doc = open(source_dir+'training_loss-'+filepath+'.txt', "w")
    doc2 = open(source_dir+'test_acc-'+filepath+'.txt', "w")
    doc3 = open(source_dir+'training_acc-' + filepath + '.txt', "w")
    check_interval=2000
    batch_number = int(6000*8/(batchsize*check_interval))
    #print(batch_number)
    training_loss_vec = [] #np.zeros(noe*check_interval)
    text_acc_vec = [] #np.zeros(noe*check_interval)
    training_acc_vec = []
    for epoch in range(noe):  # loop over the dataset multiple times
        time_begin  = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            PATH = source_folder +"betatwo"+str(beta_2)+"epoch" + str(epoch) + 'cifar_net.pth'
            torch.save(net.state_dict(), PATH)
            # print statistics
            running_loss += loss.item()
            if i % check_interval == (check_interval-1):    # print every 2000 mini-batches
                time_end = time.time()
                time_elapsed = time_end - time_begin
                time_begin = time.time()
                print('[%d, %5d] loss: %.3f, Ça coûte %.3f' %
                      (epoch + 1, i + 1, running_loss / check_interval, time_elapsed))
                training_loss_vec.append(running_loss/check_interval)
                acc = test_accuracy(net)
                t_acc = training_accuracy(net)
                text_acc_vec.append(acc)
                training_acc_vec.append(t_acc)
                print(running_loss / check_interval, file=doc)
                print(acc, file=doc2)
                print(t_acc, file=doc3)
                running_loss = 0.0
        if epoch % 1 == 0:
            for p in optimizer.param_groups:
                p['lr'] = initial_lr/np.sqrt(1+epoch)
    doc.close()
    doc2.close()
    doc3.close()

    xvar=np.arange(len(training_loss_vec))/len(training_loss_vec)*noe
    #plt.subplot(122)
    plt.figure(1)
    plt.title("Training Loss", fontsize=15)
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.plot(xvar, np.array(training_loss_vec))
    plt.savefig('training_loss'+filepath+'.png')

    #plt.subplot(122)
    plt.cla()
    plt.figure(1)
    plt.title("Test Accuracy", fontsize=15)
    plt.xlabel('epochs')
    plt.ylabel('test accuracy(%)')
    plt.plot(xvar, np.array(text_acc_vec))
    plt.savefig('test_accuracy'+filepath + '.png')

    plt.cla()
    plt.figure(1)
    plt.title("Train Accuracy", fontsize=15)
    plt.xlabel('epochs')
    plt.ylabel('test accuracy(%)')
    plt.plot(xvar, np.array(training_acc_vec))
    plt.savefig('train_accuracy'+filepath + '.png')


def test_accuracy(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def training_accuracy(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    sourcedir = './raw_data/'
    names = os.listdir(sourcedir)
    name = time.strftime("%Y-%m-%d-%H", time.localtime())
    if name not in names:
        os.mkdir(sourcedir+name)
    source_folder = sourcedir+name+"/"
    for batchsize in [16]:
        trainset = torchvision.datasets.MNIST(root='./mnist/data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                  shuffle=True, num_workers=0)

        testset = torchvision.datasets.MNIST(root='./mnist/data', train=False,
                                               download=True, transform=transform)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
        #                                         shuffle=False, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                                 shuffle=True, num_workers=0)


        for beta_2 in [0.99,0.8]:
            print("batch size:", batchsize, 'beta_2:', beta_2, 'training begins')
            # nueral net
            beta_1 = 0.1
            #beta_2 = 0.9
            args={'num_channels':1, 'num_classes':10}
            net = CNNMnist(args).to(device)

            # resnet
            #net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
            #net.eval()
            #net = net.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(beta_1, beta_2), eps=1e-08, weight_decay=0, amsgrad=False)
            print('Begin training')
            train(net, 50, source_folder, "batchsize="+str(batchsize)+"beta1="+str(beta_1)+";beta2="+str(beta_2), 0.001, batchsize, beta_2, device)
            print('Finished Training')

            del(net)

