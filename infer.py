from eval import load_model
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import argparse
import os

def Inference(device, model, img_path, out_path):
    count = 0   

    # Load image
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    model.eval()
    print(f"Found {len(os.listdir(img_path))} images")

    for img_f in os.listdir(img_path):
        ext = img_f.split(".")[-1]
        image_path = os.path.join(img_path, img_f)    
        img_array = Image.open(image_path).convert("RGB")
        # img_ls.append(img_array)
    
        temp = data_transforms(img_array).unsqueeze(dim=0)
        load = DataLoader(temp)
        
        for x in load:
            x = x.to(device)
            output = model(x)
            _, pred = torch.max(output, 1)

            # Show image
            image = cv2.imread(image_path)
            print(image_path)
            print(f"Class: {pred}")
            if (pred[0] == 1): 
                print("Predicted =====> Positive")
                cv2.imwrite(f"{out_path}/Pos{count}.{ext}", image)
            else: 
                print("Predicted =====> Negative")
                cv2.imwrite(f"{out_path}/Neg{count}.{ext}", image)
            count += 1
            print()
            
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default= os.getcwd()+"/saved_model/pretrained.pt", help="Trained model path")
    parser.add_argument("--img_path", type=str, default=os.getcwd()+"/infer_images", help="Image folder path")
    parser.add_argument("--out_path", type=str, default=os.getcwd()+"/output", help="Predicted Image folder path")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    args = parse_opt()
    model = load_model(device, args.model_path)
    
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    Inference(device, model, args.img_path, args.out_path)

    print(f"Predicted images are saved in: {args.out_path}")