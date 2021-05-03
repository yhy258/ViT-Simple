import torch
from torchvision import datasets, transforms

def my_Cifar10():
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.RandomCrop(224, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)
                                    ])

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(224), transforms.Normalize(mean, std)])

    train_dataset = datasets.CIFAR10(
        root='./.data',
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.CIFAR10(
        root='./.data',
        train=False,
        transform=transform_test,
        download=True
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 16, True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 16, True)

    return train_dataset, test_dataset, train_dataloader, test_dataloader