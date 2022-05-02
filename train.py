import os
import torch
import label_utils
import torchvision
import datetime
from data_loader import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import gdown
import tarfile


def get_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 4  # 3 class (juice, soda, water) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__== "__main__" :

    url = "https://drive.google.com/file/d/1AdMbVK110IKLG7wJKhga2N2fitV1bVPA/view?usp=sharing"
    output = 'drinks.tar.gz'
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)

    tar = tarfile.open(output)
    tar.extractall()
    tar.close()

    train_dict, train_classes = label_utils.build_label_dictionary(
        "drinks/labels_train.csv")
    test_dict, test_classes = label_utils.build_label_dictionary(
        "drinks/labels_test.csv")

    train_split = ImageDataset(train_dict, get_transform(train=True))
    test_split = ImageDataset(test_dict, get_transform(train=False))

    data_loader = DataLoader(train_split,
                            batch_size=8,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(test_split,
                            batch_size=8,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=utils.collate_fn)    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 4

    # get the model using our helper function
    model = get_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.05,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    
    # let's train it for 10 epochs
    from torch.optim.lr_scheduler import StepLR
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    
    torch.save(model, 'model.pth')