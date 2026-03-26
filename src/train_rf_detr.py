if __name__ == "__main__":
    from rfdetr import RFDETRBase, RFDETRLarge
    import os
    import torch
    import random
    import numpy as np

    # ================= CONFIG =================
    dataset_dir = r"D:\Projects\Project1_wheatheads\project1_dataset\images"  # your dataset root
    # Choose model:
    model = RFDETRBase()       # Use Base model
    # model = RFDETRLarge()    # Uncomment to use Large model

    # ================= SEED (matches YOLO seed=0) =================
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ================= YOLO-MATCHED HYPERPARAMETERS =================
    epochs = 300               # epochs=300
    batch_size = 4             # batch=4
    lr = 8e-4                  # lr0=0.0008
    img_size = 560             # imgsz=640
    patience = 25              # patience=25
    warmup_epochs = 3          # warmup_epochs=3

    # YOLO augmentations (matching your exact values)
    flip_prob = 0.5            # fliplr=0.5, flipud=0.0
    scale = 0.2                # scale=0.2
    rotate = 2.0               # degrees=2.0
    translate = 0.1            # translate=0.1
    # mosaic=0.0, mixup=0.0, copy_paste=0.0 → disabled by default

    # ================= SINGLE TRAINING CALL (handles everything) =================
    print("🚀 Starting RF-DETR training with YOLO-matched settings...")
    
    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        resolution=img_size,           # imgsz=640
        multi_scale=False, 
        expanded_scales=False,
        early_stopping=True,
        early_stopping_patience=patience,  # patience=25
        warmup_epochs=warmup_epochs,   # warmup_epochs=3
        flip_prob=flip_prob,           # fliplr=0.5
        scale=scale,                   # scale=0.2
        degrees=rotate,                # degrees=2.0
        translate=translate,           # translate=0.1
        # No need for hsv/perspective/shear - use RF-DETR defaults or omit
        seed=seed,                     # if supported
        workers=4,                     # num_workers equivalent
        save_dir=r"C:\RF-DETR\checkpoints",       # save best model here
        save_period=5
    )

    print(f"✅ Training finished for model: {type(model).__name__}")
    print("📁 Check 'checkpoints' folder for best model weights.")
