import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ICDAR15Dataset(Dataset):
    def __init__(self, img_dir, gt_dir, img_size=640, ignore_unreadable=True):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        self.ignore_unreadable = ignore_unreadable
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']
        self.img_files = []
        for ext in extensions:
            self.img_files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_files = sorted(self.img_files)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomApply([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            ], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        image = self.transform(image)
        
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        gt_name = f"gt_{name_no_ext}.txt"
        gt_path = os.path.join(self.gt_dir, gt_name)
        if not os.path.exists(gt_path):
            gt_path = os.path.join(self.gt_dir, name_no_ext + ".txt")

        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        if os.path.exists(gt_path):
            with open(gt_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 8:
                        try:
                            coords = [float(x.strip().replace('\ufeff', '')) for x in parts[:8]]
                            poly = np.array(coords).reshape(-1, 2)
                            poly[:, 0] = poly[:, 0] * (self.img_size / w)
                            poly[:, 1] = poly[:, 1] * (self.img_size / h)
                            import cv2
                            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                        except ValueError:
                            pass
        
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask
