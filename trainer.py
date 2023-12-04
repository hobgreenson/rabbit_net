import argparse
import logging
import os
import time
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from device import DEVICE
from model import build_model, freeze_weights


ImageFile.LOAD_TRUNCATED_IMAGES = True


class RabbitDataset(Dataset):
    pos_path = "/content/drive/MyDrive/rabbitnet/data/rabbits"
    neg_path = "/content/drive/MyDrive/rabbitnet/data/non_rabbits"
    test_pos_path = "/content/drive/MyDrive/rabbitnet/data/test_rabbits"
    test_neg_path = "/content/drive/MyDrive/rabbitnet/data/test_non_rabbits"

    def __init__(self, preprocess, train=True):
        self.data = []
        self.labels = []
        self.paths = []

        if train:
            pos_path = self.pos_path
            neg_path = self.neg_path
        else:
            pos_path = self.test_pos_path
            neg_path = self.test_neg_path

        for label, path in [(1.0, pos_path), (0.0, neg_path)]:
            for fn in os.listdir(path):
                if fn.lower().endswith(".jpg"):
                    full_path = "/".join([path, fn])
                    with Image.open(full_path) as im:
                        self.data.append(preprocess(im))
                    self.labels.append(label)
                    self.paths.append(full_path)

        self.data = torch.stack(self.data, dim=0)
        self.labels = torch.tensor(self.labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def time_loader(dataloader, n=10):
    timings = []
    for _ in range(n):
        start = time.perf_counter()
        for _ in enumerate(dataloader):
            pass
        timings.append(time.perf_counter() - start)
    logging.info(sum(timings) / n)


def train_one_epoch(model, optimizer, loss_fn, dataloader):
    model.train()
    running_loss = 0.0
    for i, (x, y) in enumerate(dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        yhat = model(x)
        loss = loss_fn(yhat.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / (i + 1)
    return avg_loss


def validate(model, loss_fn, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            yhat = model(x)
            loss = loss_fn(yhat.squeeze(), y.squeeze())
            running_loss += loss.item()
    avg_loss = running_loss / (i + 1)
    return avg_loss


def main(epochs, batch_size, learning_rate, checkpoint_interval):
    checkpoint_dir = "/content/drive/MyDrive/rabbitnet/checkpoints"
    model, preprocess = build_model()
    model = freeze_weights(model)
    model = model.to(DEVICE)

    train_dataset = RabbitDataset(preprocess, train=True)
    val_dataset = RabbitDataset(preprocess, train=False)
    batch_size = len(train_dataset) if batch_size < 0 else batch_size

    logging.info(
        f"Training with epochs={epochs}; batch_size={batch_size}; learning_rate={learning_rate}"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loss_log = []
    val_loss_log = []
    for t in range(epochs):
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_dataloader)
        train_loss_log.append(train_loss)
        val_loss = validate(model, loss_fn, val_dataloader)
        val_loss_log.append(val_loss)

        if (t + 1) % checkpoint_interval == 0:
            checkpoint_path = "/".join([checkpoint_dir, f"rabbit_net_{t + 1}.pt"])
            torch.save(
                {
                    "epoch": t + 1,
                    "batch_size": len(train_dataset),
                    "learning_rate": learning_rate,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss_log,
                    "valid_loss": val_loss_log,
                },
                checkpoint_path,
            )
            logging.info(
                f"Epoch {t + 1}; Train/Valid Loss {train_loss}/{val_loss}; Saved to: {checkpoint_path}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=300, help="number of epochs to train for")
    parser.add_argument(
        "--learning_rate", default=1e-3, help="learning rate to pass to the optimizer"
    )
    parser.add_argument(
        "--batch_size", default=-1, help="batch size to use during training"
    )
    parser.add_argument(
        "--checkpoint_interval", default=50, help="number of epochs between checkpoint saves"
    )
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.learning_rate, args.checkpoint_interval)
