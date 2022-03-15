import loader
import torch
import torchvision
import torch.nn as nn
import os
import argparse

def load_model(device, model_path):
    # Load checkpoint
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 2, bias=True)
    print("----> Loading checkpoint")
    checkpoint = torch.load(model_path, map_location=device) # Try to load last checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded")
    model.to(device)
    
    return model

def test(device, model, loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, pred = torch.max(output, 1)
            correct += (pred == y).sum().item()
            test_loss += criterion(output, y)    

    test_loss /= len(loader.dataset)
    print(f"Average loss: {test_loss}   Accuracy: {correct} / {len(loader.dataset)}  {int(correct) / len(loader.dataset) * 100}%")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default= os.getcwd()+"/saved_model/pretrained.pt", help="Trained model path")
    parser.add_argument("--path", type=str, default= os.getcwd()+"/dataset", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=64, help="Data batch-size")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_opt()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    data_loader = loader.data_loader(args.path, args.batch_size)
    model = load_model(device, args.model_path)
    print(f"---------------------------")
    print(f"Found {len(data_loader['test'].dataset)} for evaluation")
    print(f"---------------------------")
    test(device, model, data_loader['test'], criterion)