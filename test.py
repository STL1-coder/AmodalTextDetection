import argparse
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader
from model import OcclusionAwareDetector
from dataset import ICDAR15Dataset
from engine import Trainer

def run_icdar_inference(trainer, img_path, gt_dir, threshold=0.5):
    trainer.model.eval()
    try:
        orig_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {img_path}: {e}")
        return

    w_orig, h_orig = orig_img.size
    img_np = np.array(orig_img)
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    inp = transform(orig_img).unsqueeze(0).to(trainer.device)
    
    with torch.no_grad():
        logits = trainer.model(inp)
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    prob_map_resized = cv2.resize(prob_map, (w_orig, h_orig))
    img_gt = img_np.copy()
    filename = os.path.basename(img_path)
    name_no_ext = os.path.splitext(filename)[0]
    gt_name = f"gt_{name_no_ext}.txt"
    gt_path = os.path.join(gt_dir, gt_name)
    if not os.path.exists(gt_path):
        gt_path = os.path.join(gt_dir, name_no_ext + ".txt")

    if os.path.exists(gt_path):
        with open(gt_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    try:
                        coords = [float(x.strip().replace('\ufeff', '')) for x in parts[:8]]
                        poly = np.array(coords, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(img_gt, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
                    except ValueError:
                        pass
    else:
        cv2.putText(img_gt, "No GT File Found", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    heatmap = (prob_map_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    binary_mask = (prob_map_resized > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_pred = img_np.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box) 
            cv2.drawContours(img_pred, [box], 0, (0, 255, 0), 2)

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title("Input Image")
    plt.imshow(orig_img)
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.title("Ground Truth (Blue)")
    plt.imshow(img_gt)
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.title("Heatmap")
    plt.imshow(overlay_rgb)
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.title(f"Prediction (Green)\nThresh: {threshold}")
    plt.imshow(img_pred)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Test OCR Segmentation Model")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to test images")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to test ground truth")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, choices=['eval', 'viz'], default='eval', help="eval: metrics, viz: plot images")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device set to: {device}")

    model = OcclusionAwareDetector()
    trainer = Trainer(model, device)
    
    if os.path.exists(args.checkpoint):
        trainer.load_checkpoint(args.checkpoint)
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    test_ds = ICDAR15Dataset(args.img_dir, args.gt_dir)
    
    if len(test_ds) == 0:
        print("No images found.")
        return

    if args.mode == 'eval':
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loss, test_iou, test_prec, test_rec, test_f1 = trainer.evaluate(test_loader)
        print("\n" + "="*40)
        print("       FINAL TEST RESULTS       ")
        print("="*40)
        print(f"Average Loss:      {test_loss:.4f}")
        print(f"Pixel IoU:         {test_iou:.4f}")
        print(f"Pixel Precision:   {test_prec:.4f}")
        print(f"Pixel Recall:      {test_rec:.4f}")
        print(f"Pixel F1-Score:    {test_f1:.4f}")
        print("="*40)
    
    elif args.mode == 'viz':
        import random
        indices = random.sample(range(len(test_ds)), min(5, len(test_ds)))
        for idx in indices:
            test_img_path = test_ds.img_files[idx]
            print(f"Visualizing: {os.path.basename(test_img_path)}")
            run_icdar_inference(trainer, test_img_path, test_ds.gt_dir)

if __name__ == "__main__":
    main()
