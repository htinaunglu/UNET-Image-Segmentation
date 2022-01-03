import torch
import sys
import albumentations as A
import os
import argparse
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    get_loaders,
    save_predictions_as_imgs,
)
import smdebug.pytorch as smd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_fn(loader, model, optimizer, loss_fn , scaler, hook ):
    loop = tqdm(loader)
    hook.set_mode(smd.modes.TRAIN)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
def validate(val_loader, model, loss_fn, hook):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    running_loss = 0
    model.eval()
    hook.set_mode(smd.modes.EVAL)

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device=DEVICE)
            y = y.to(device=DEVICE).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            loss = loss_fn(preds, y)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            accuracy = num_correct/num_pixels*100
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss // len(val_loader)
        logger.info(
            "\nTest set: Average loss: {:.4f}\n".format(avg_loss)
        )
                    
        logger.info(
            "\nTest set: Accuracy: {:.4f}\n".format(accuracy)
        )

    logger.info(
        f"Got {num_correct}/{num_pixels} with Accuracy {accuracy:.2f}"
    )
    logger.info(f"Dice score: {dice_score/len(val_loader)}")
    return accuracy
        
def main(args):
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    NUM_WORKERS = 2 # the number of processes that generate batches in parallel
    IMAGE_HEIGHT = 160  # 1280 originally
    IMAGE_WIDTH = 240  # 1918 originally
    PIN_MEMORY = True
    
    TRAIN_IMG_DIR = args.train
    TRAIN_MASK_DIR = args.train_masks
    VAL_IMG_DIR = args.val
    VAL_MASK_DIR = args.val_masks
    
    
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    hook.register_loss(loss_fn)
    
    scaler = torch.cuda.amp.GradScaler()
    val_acc =[]
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, hook)
        valid_accuracy = validate(val_loader, model, loss_fn , hook)
        # save model
        torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
        save_predictions_as_imgs(
        val_loader, model, folder='/opt/ml/output', device=DEVICE
        )
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() # adding the args parsers to use with the notebook estimator call

    
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 64,
        metavar = "N",
        help = "input batch size for training (default: 64)",
    )
    
    parser.add_argument(
        "--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--num_epochs", type = int, default = 20, metavar = "Epochs", help = "number of epochs (default: 20)"
    )

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument('--train_masks', type=str, default=os.environ['SM_CHANNEL_TRAIN_MASKS'])
    parser.add_argument('--val_masks', type=str, default=os.environ['SM_CHANNEL_VAL_MASKS'])

    args, _ = parser.parse_known_args()
    print(args)
    main(args)
