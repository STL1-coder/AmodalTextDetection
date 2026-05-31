import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, device, lr=1e-4, checkpoint_dir='./checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()
        self.criterion = nn.BCEWithLogitsLoss()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch} Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        total_iou = 0
        total_prec = 0
        total_rec = 0
        total_f1 = 0
        
        with torch.no_grad():
            for images, masks in tqdm(loader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                masks_bool = masks > 0.5
                
                intersection = (preds & masks_bool).float().sum()
                union = (preds | masks_bool).float().sum()
                iou = (intersection + 1e-6) / (union + 1e-6)
                
                tp = intersection
                fp = (preds & ~masks_bool).float().sum()
                fn = (~preds & masks_bool).float().sum()
                
                precision = (tp + 1e-6) / (tp + fp + 1e-6)
                recall = (tp + 1e-6) / (tp + fn + 1e-6)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
                
                total_iou += iou.item()
                total_prec += precision.item()
                total_rec += recall.item()
                total_f1 += f1.item()

        n = len(loader)
        return total_loss / n, total_iou / n, total_prec / n, total_rec / n, total_f1 / n

    def save_model(self, epoch):
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), path)
        latest_path = os.path.join(self.checkpoint_dir, "latest_model.pth")
        torch.save(self.model.state_dict(), latest_path)
        print(f"Saved model to {path}")

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded checkpoint from {path}")
