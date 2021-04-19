import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time

import os
import time

import copy

import requests
import zipfile

def test_func(v):
    print("test func triggered")
    print(v)

def  download_data(path="https://github.com/polyrand/strive-ml-fullstack-public/blob/main/06_org_documentation_scripting/data.zip?raw=true"):
    if "data.zip" not in os.listdir():
        r = requests.get(path)
        print("downloading data")
        with open("data.zip", "wb") as f:
            f.write(r.content)
    else:
        print("data already downloaded")

def extract_zip_file_data():
    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall("data")

def build_dataloader(dir="data/hymenoptera_data"):
    data_transforms = {"train": transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ]),
        "val": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ]), }

    data_dir = dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ["train", "val"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes



    return  dataloaders, dataset_sizes, class_names

def train_model(num_epochs=10, dir="data/hymenoptera_data", sel="sgd"):
    model, criterion, optimizer, scheduler, device = make_model(sel)
    since = time.time()
    dataloaders, dataset_sizes, class_names = build_dataloader(dir)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "best_model.pth")
    return model

def make_model(sel="sgd"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_conv = torchvision.models.resnet18(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = update_optimiser(sel, model=model_conv) #optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    #optimizer_conv = optim.Adam(
    #    model_conv.fc.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01, esp=1e-5
    #)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=3, gamma=0.1)

    return model_conv, criterion, optimizer_conv, exp_lr_scheduler, device

def update_optimiser(sel, model):

    if sel == "adam":
        optimizer_conv  = optim.Adam(model.fc.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01, esp=1e-5)
        return optimizer_conv
    elif sel == "sgd":
        optimizer_conv = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
        return optimizer_conv


def train_the_model(dir="data/hymenoptera_data", sel="sgd", num_epochs=10, save=False, path="model.pkl"):
    start = time.perf_counter()
    model = train_model( num_epochs, dir=dir, sel=sel)
    total = time.perf_counter() - start
    print(f"Total time {total}")
    if save:
        save_model(model,path)
    return model

def save_model(model, path="model.pkl"):
    torch.save(model, path)
