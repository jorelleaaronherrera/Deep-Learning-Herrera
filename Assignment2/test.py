import os
import torch
import torchvision
import label_utils
import utils
from data_loader import ImageDataset
from torch.utils.data import DataLoader
from train import get_model, get_transform
from engine import evaluate
import gdown
import tarfile

if __name__ == "__main__":

    url = "https://drive.google.com/file/d/1AdMbVK110IKLG7wJKhga2N2fitV1bVPA/view?usp=sharing"
    output = 'drinks.tar.gz'
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)

    tar = tarfile.open(output)
    tar.extractall()
    tar.close()

    url = "https://drive.google.com/file/d/1NDbgawtmajcrqof0GjV3ZFcfwBJdfIER/view?usp=sharing"
    output = 'model.pth'
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)

    test_dict, test_classes = label_utils.build_label_dictionary(
        "drinks/labels_test.csv")

    test_split = ImageDataset(test_dict, get_transform(train=False))

    data_loader_test = DataLoader(test_split,
                            batch_size=8,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 4

    model = get_model(num_classes)

    model = torch.load('model.pth')
    evaluate(model, data_loader_test, device=device)
