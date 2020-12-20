
'''
Script to train and test litter classifier.
'''
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Train and test network for classification task')
    parser.add_argument('--data_img',
                        help='path to directory with subdirectories with images',
                        type=str)
    parser.add_argument('--out',
                        help='path to main directory with checkpoints',
                        type=str)
    parser.add_argument('--mode', default='train',
                        help='type of procedure: test or train',
                        choices=['train', 'test'],
                        type=str)
    parser.add_argument('--name', default='test.png',
                        help='path to save test images', type=str)
    parser.add_argument('--num', help='number of images to display',
                        default=5, type=int)
    parser.add_argument('--device', help='specify device to use',
                        default="cuda:0", type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    return parser


def make_weights_for_balanced_classes(images, nclasses=6):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i]) if count[i] != 0 else N
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def load_split_train_test(datadir, resize_img=(224, 224),
                          valid_size=.3, batch_size=300,
                          mode='balanced'):
    train_transforms = transforms.Compose([transforms.Resize(resize_img),
                                           transforms.ToTensor(),
                                           ])
    train_data = torchvision.datasets.ImageFolder(datadir,
                                                  transform=train_transforms)
    num_imgs = len(train_data)
    indices = list(range(num_imgs))
    split = int(np.floor(valid_size * num_imgs))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    if mode == 'balanced':
        # For unbalanced dataset we create a weighted sampler
        train_samples = []
        for i in train_idx:
            train_samples.append(train_data.imgs[i])
        weights = make_weights_for_balanced_classes(train_samples,
                                                    len(train_data.classes))
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            torch.DoubleTensor(weights), len(weights))
        test_samples = []
        for i in test_idx:
            test_samples.append(train_data.imgs[i])
        weights = make_weights_for_balanced_classes(test_samples,
                                                    len(train_data.classes))
        test_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            torch.DoubleTensor(weights), len(weights))
    else:
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler,
                                              batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(train_data,
                                             sampler=test_sampler,
                                             batch_size=batch_size)
    return trainloader, testloader


def predict_image(image, model, device):
    image_tensor = test_transforms(image).float().to(device)
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor)
    #input_img = input_img.to(device)
    model.to(device)
    output = model(input_img)
    index = output.data.cpu().numpy().argmax()
    return index


def get_random_images(data_dir, test_transforms, num=10):
    data = torchvision.datasets.ImageFolder(data_dir,
                                            transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data,
                                         sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels, classes


def eval(testloader, model, device):
    model.to(device)
    model.eval()
    print('Evaluation...')
    classes = testloader.dataset.classes
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(f'Correct: {class_correct}\nTotal: {class_total}')
    for i in range(len(classes)):
        print('Accuracy of %5s : %.2f %%' % (
            classes[i],
            (100 * class_correct[i] / class_total[i]
             if class_total[i] != 0 else 0)))


def train(args, model, device):
    # load data
    trainloader, testloader = load_split_train_test(args.data_img)

    # use a Classification Cross-Entropy loss and SGD with momentum.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.fc.parameters(),
        lr=args.lr,
        momentum=args.momentum)

    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(
                            torch.FloatTensor)).item()
                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(testloader))
                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                running_loss = 0
                model.train()
    # plotting loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(args.out, 'logs.jpg'))

    print('Finished Training...')
    model_name = 'classification_net.pth'
    out_path = os.path.join(args.out, 'classification_net.pth')
    torch.save(model.state_dict(), out_path)

    eval(testloader, model, device)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    # define a Convolutional Neural Network object
    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    print(device)
    model = torchvision.models.resnet50(pretrained=True).to(device)
    print(model)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 10),
                             nn.LogSoftmax(dim=1))

    if args.mode == 'test':
        to_pil = transforms.ToPILImage()
        test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              ])
        model.load_state_dict(
            torch.load(
                os.path.join(
                    args.out,
                    'classification_net.pth')))
        model.eval()
        images, labels, gt_cl = get_random_images(args.data_img,
                                                  test_transforms, args.num)
        classes = ['Bio', 'Glass', 'Metals-and-plastics',
                   'Non-recyclable', 'Other', 'Paper']
        fig = plt.figure(figsize=(10, 10))
        for ii in range(len(images)):
            image = to_pil(images[ii])
            index = predict_image(image, model, device)
            sub = fig.add_subplot(len(images), 1, ii + 1)
            res = int(labels[ii]) == index
            sub.set_title("GT: " + str(gt_cl[labels[ii]]) +
                          ", Pred: " + str(classes[index]))
            plt.axis('off')
            plt.imshow(image)
        plt.savefig(os.path.join(args.out, args.name))
    else:
        train(args, model, device)
