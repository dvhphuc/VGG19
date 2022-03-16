import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def data_loader(path, batch_size):
    preprocess = {'train': transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]),
                  'valid': transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]),
                  'test': transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        
    }

    dataset = {x: ImageFolder(os.path.join(path, x), preprocess[x]) for x in ['train', 'valid', 'test']}
    loader = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid', 'test']}

    return loader