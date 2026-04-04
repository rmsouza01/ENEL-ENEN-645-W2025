import os
import glob
import torch
import wandb
import numpy as np
from tqdm import tqdm

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityd, RandCropByPosNegLabeld, EnsureTyped,
    SaveImaged, AsDiscrete, ThresholdIntensityd, Lambdad
)
# REMOVED: from monai.handlers.utils import write_nifti (Deprecated)
# REMOVED: from monai.handlers.utils import from_engine (Not used in this script)

# --- Configuration ---
WANDB_PROJECT = "3D_Medical_Segmentation"
DATA_ROOT = "/work/TALC/ensf617_2026w/Skull-stripping" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_dicts(folder):
    """Globs images and labels from a specific folder."""
    images = sorted(glob.glob(os.path.join(DATA_ROOT, "Images", folder,  "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(DATA_ROOT, "Masks", folder,  "*.nii.gz")))
    return [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

# --- 1. Initialize W&B ---
wandb.init(project=WANDB_PROJECT, config={
    "learning_rate": 1e-3,
    "epochs": 15,
    "batch_size": 2,
    "roi_size": (128, 128, 128),
})
config = wandb.config

# --- 2. Transforms & DataLoaders ---
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    Lambdad(keys=["label"], func=lambda x: (x > 0.5).float()),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=config.roi_size, pos=1, neg=1, num_samples=4
    ),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    Lambdad(keys=["label"], func=lambda x: (x > 0.5).float()),
    EnsureTyped(keys=["image", "label"]),
])

train_ds = Dataset(data=get_data_dicts("Train"), transform=train_transforms)
val_ds = Dataset(data=get_data_dicts("Val"), transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

# --- 3. Model, Loss, Metric ---
model = UNet(
    spatial_dims=3, in_channels=1, out_channels=1,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
).to(device)

loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
dice_metric = DiceMetric(include_background=True, reduction="mean")

# --- 4. Training Loop ---
best_metric = -1

for epoch in range(config.epochs):
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / step
    wandb.log({"train/loss": avg_loss, "epoch": epoch})

    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            val_outputs = sliding_window_inference(val_inputs, config.roi_size, 4, model)
            
            # Post-processing
            val_outputs = [torch.where(torch.sigmoid(i) > 0.5, 1, 0) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=decollate_batch(val_labels))

        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        wandb.log({"val/dice": metric, "epoch": epoch})

        # Visualization
        sample_img = val_inputs[0, 0, :, :, 20].cpu().numpy() 
        sample_gt = val_labels[0, 0, :, :, 20].cpu().numpy()
        sample_pred = val_outputs[0][0, :, :, 20].cpu().numpy()
        
        wandb.log({
            "val_vis": wandb.Image(sample_img, masks={
                "prediction": {"mask_data": sample_pred, "class_labels": {0: "bg", 1: "target"}},
                "ground_truth": {"mask_data": sample_gt}
            })
        })

        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), "best_model2.pth")
            wandb.save("best_model2.pth")

wandb.finish()

# --- 5. Final Testing ---
test_dice_metric = DiceMetric(include_background=True, reduction="mean")
test_ds = Dataset(data=get_data_dicts("Test"), transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)

post_pred = Compose([AsDiscrete(threshold=0.5)])
save_transform = SaveImaged(
    keys="pred", 
    output_dir="./test_results", 
    output_postfix="seg", 
    resample=False # Use False if you didn't change spacing in transforms
)

model.load_state_dict(torch.load("best_model2.pth"))
model.eval()

test_table = wandb.Table(columns=["Filename", "Dice_Score"])

print("Running Final Test Inference...")
with torch.no_grad():
    for test_data in tqdm(test_loader):
        test_inputs = test_data["image"].to(device)
        # MOVE LABELS TO DEVICE HERE:
        test_labels_device = test_data["label"].to(device)
        
        filename = os.path.basename(test_data["image"].meta["filename_or_obj"][0])
        test_outputs = sliding_window_inference(test_inputs, config.roi_size, 4, model)
        
        # Use the device-resident labels for the metric
        labels = decollate_batch(test_labels_device)
        preds = [post_pred(torch.sigmoid(i)) for i in decollate_batch(test_outputs)]
        
        test_dice_metric(y_pred=preds, y=labels)
        
        current_dice = test_dice_metric.aggregate().item()
        test_table.add_data(filename, current_dice)
        
        # Prepare for Saving
        test_data["pred"] = test_outputs
        
        for d in decollate_batch(test_data):
            d["pred"] = post_pred(torch.sigmoid(d["pred"])) 
            save_transform(d)

final_test_dice = test_dice_metric.aggregate().item()
test_dice_metric.reset()

wandb.log({
    "test/mean_dice": final_test_dice,
    "test/per_volume_results": test_table
})

print(f"Final Test Mean Dice: {final_test_dice:.4f}")
wandb.finish()
