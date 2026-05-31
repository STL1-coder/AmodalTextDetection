import argparse
import torch
from torch.utils.data import DataLoader
from model import OcclusionAwareDetector
from dataset import ICDAR15Dataset
from engine import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train OCR Segmentation Model")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to training images")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to training ground truth")
    parser.add_argument("--test_img_dir", type=str, required=False, help="Path to test images")
    parser.add_argument("--test_gt_dir", type=str, required=False, help="Path to test ground truth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device set to: {device}")

    train_ds = ICDAR15Dataset(args.img_dir, args.gt_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    test_loader = None
    if args.test_img_dir and args.test_gt_dir:
        test_ds = ICDAR15Dataset(args.test_img_dir, args.test_gt_dir)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = OcclusionAwareDetector()
    trainer = Trainer(model, device, lr=args.lr, checkpoint_dir=args.checkpoint_dir)

    print(f"Starting Training for {args.epochs} epochs...")
    for ep in range(args.epochs):
        current_epoch = ep + 1
        train_loss = trainer.train_one_epoch(train_loader, current_epoch)
        
        if test_loader:
            test_loss, test_iou, test_prec, test_rec, test_f1 = trainer.evaluate(test_loader)
            print("-" * 80)
            print(f"Epoch {current_epoch} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
            print(f"Metrics: IoU: {test_iou:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | F1: {test_f1:.4f}")
            print("-" * 80)
        else:
            print(f"Epoch {current_epoch} Summary | Train Loss: {train_loss:.4f}")
            
        trainer.save_model(current_epoch)

if __name__ == "__main__":
    main()
