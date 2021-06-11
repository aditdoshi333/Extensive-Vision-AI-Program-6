import torch
from torchvision import datasets, transforms


def train_test_data_loader(batch_size):

    # Train and test dataset
    train_set = datasets.MNIST('./data', 
                   train=True, 
                   download=True,
                   transform=transforms.Compose([
                                       transforms.RandomRotation((-8, 8), fill=(1,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) 
                                       ]))


    test_set = datasets.MNIST('./data', 
                      train=False, 
                      download=True,
                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                          ]))
    # Checking GPU is available or not
    SEED = 1
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    # Data loader args that will be passed to dataloader function
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True,
                                                                                                           batch_size=64)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

    return train_loader, test_loader

