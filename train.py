
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

from PIL import Image
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import shutil, argparse
import matplotlib.pyplot as plt

def get_dataloaders(data_dir):
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'

    ### SET TRAIN LOADER
    train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    ### SET VALID AND TEST LOADER
    test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    validloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return trainloader, validloader, testloader

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def save_checkpoint(state, save_dir, is_best=False, filename='checkpoint.pth.tar'):
    path = save_dir + filename
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, 'model_best.pth.tar')


def train_model(trainloader, validloader, arch, hidden_units, learning_rate, \
    cuda, epochs, save_dir, save_every):
    # Initial parameters
    print_every = 1
    save_every = 50

    # Get model
    model = eval("models.{}(pretrained=True)".format(arch))

    # Freeze the feature parameters
    for params in model.parameters():
        params.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, hidden_units)),
    ('relu', nn.ReLU()), 
    ('fc2', nn.Linear(hidden_units, 102)),
    ('drop', nn.Dropout(p=0.5)),
    ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    if cuda:
        model.cuda()
    else:
        model.cpu()
        
    epochs = epochs
    steps = 0

    for e in range(epochs):
        model.train()
        running_loss = 0
        accuracy_train = 0
        
        for images, labels in iter(trainloader):
            steps += 1
            
           # print("Step number {}".format(steps))

            inputs, labels = Variable(images), Variable(labels)
            
            optimizer.zero_grad()
            
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            
            #print("Performing forward")
            output = model.forward(inputs)
            #print("Criterion")
            loss = criterion(output, labels)
            #print("Performing backward")
            loss.backward()
            #print("Step of the optimizer")
            optimizer.step()
            
            running_loss += loss.item()
            ps_train = torch.exp(output).data
            equality_train = (labels.data == ps_train.max(1)[1])
            accuracy_train += equality_train.type_as(torch.FloatTensor()).mean()
            
            
            
            if steps % print_every == 0:
                model.eval()
                
                accuracy = 0
                valid_loss = 0
                
                for images, labels in validloader:
                    with torch.no_grad():
                        inputs = Variable(images)
                        labels = Variable(labels)

                        if cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output = model.forward(inputs)

                        valid_loss += criterion(output, labels).item()

                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])

                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs), 
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}..".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0
                model.train()
                
            if steps % save_every == 0:
                print("Saving step number {}...".format(steps))
                state = {'state_dict': model.classifier.state_dict(),
                         'optimizer' : optimizer.state_dict(),
                         'class_to_idx':train_dataset.class_to_idx}
                
                save_checkpoint(state, save_dir)
                print("Done!")



    return model

def main():
    # Some initial parameters
    test_loaders=False

    # Command line arguments
    parser = argparse.ArgumentParser(description='Train a new network on a data set')

    parser.add_argument('data_dir', type=str, \
        help='Path of the Image Dataset (with train, valid and test folders)')
    parser.add_argument('--save_dir', type=str, \
        help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, \
        help='Models architeture. Default is densenet121. Choose one at https://pytorch.org/docs/master/torchvision/models.html')
    parser.add_argument('--learning_rate', type=float, \
        help='Learning rate. Default is 0.01')
    parser.add_argument('--hidden_units', type=int, \
        help='Hidden units. Default is 200')
    parser.add_argument('--epochs', type=int, \
        help='Number of epochs. Default is 3')
    parser.add_argument('--gpu', action='store_true', \
        help='Use GPU for inference if available')
    parser.add_argument('--save_every', type=int, \
        help='Number of steps to save the checkpoint. Default is 50')
    
    args, _ = parser.parse_known_args()

    data_dir = args.data_dir

    save_dir = './'
    if args.save_dir:
        save_dir = args.save_dir

    arch = 'densenet121'
    if args.arch:
        arch = args.arch

    learning_rate = 0.01
    if args.learning_rate:
        learning_rate = args.learning_rate

    hidden_units = 200
    if args.hidden_units:
        hidden_units = args.hidden_units

    epochs = 3
    if args.epochs:
        epochs = args.epochs

    save_every = 50
    if args.save_every:
        save_every = args.save_every

    cuda = False
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
        else:
            print("Warning! GPU flag was set however no GPU is available in \
                the machine")

    trainloader, validloader, testloader = get_dataloaders(data_dir)

    # Test loaders
    if test_loaders:
        images, labels = next(iter(trainloader))
        imshow(images[2])
        plt.show()

        images, labels = next(iter(validloader))
        imshow(images[2])
        plt.show()

        images, labels = next(iter(testloader))
        imshow(images[2])
        plt.show()

    train_model(trainloader, validloader, arch=arch, hidden_units=hidden_units,\
     learning_rate=learning_rate, cuda=cuda, epochs=epochs, save_dir=save_dir, \
     save_every=save_every)







if __name__ == '__main__':
    main()