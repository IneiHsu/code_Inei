import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# from src.datasets import ThingsMEGDataset
# from src.models import BasicConvClassifier
from src.utils import set_seed


# def preprocess_data(X, resample_rate=250, lowcut=0.5, highcut=40.0, fs=1000.0):
#     # # Calculate baseline: using the first 20% of the data as an example
#     # baseline = torch.mean(X[:, :, :int(0.2 * X.shape[2])], dim=2, keepdim=True)
#     seq_len = X.shape[-1]
#     baseline_window = int(0.2 * seq_len)
#     baseline = torch.mean(X[:, :, :baseline_window], dim=-1, keepdim=True)
#     # # Subtract baseline from data to correct it
#     X_pre = X - baseline
#
#     # Convert to float32 (if necessary)
#     # X_pre = X_pre.astype(np.float32)
#
#
#     return X_pre
#
# def preprocess_dataset(dataset, has_subject_idxs=True):
#     processed_data = []
#     # for data in dataset:
#     for i in range(len(dataset)-1):
#         if has_subject_idxs:
#             X, y, subject_idxs = dataset[i]
#             X = preprocess_data(X)
#             processed_data.append((X, y, subject_idxs))
#         else:
#             X, y = dataset[i]
#             X = preprocess_data(X)
#             processed_data.append((X, y))
#     return processed_data

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    print("Loading train set...")
    train_set = ThingsMEGDataset("train", args.data_dir)
    # train_set = preprocess_dataset(train_set, has_subject_idxs=True)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)

    print(f"Train set loaded with {len(train_set)} samples")
    print("Train set sample:", train_set[0])
    print("num_classes",train_set.num_classes)
    print("seq_len",train_set.seq_len)
    print("num_channels",train_set.num_channels)


    val_set = ThingsMEGDataset("val", args.data_dir)
    # val_set = preprocess_dataset(val_set, has_subject_idxs=True)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)

    print(f"Valid set loaded with {len(val_set)} samples")
    print("Valid set sample:", val_set[0])
    print("num_classes", val_set.num_classes)
    print("seq_len", val_set.seq_len)
    print("num_channels", val_set.num_channels)

    test_set = ThingsMEGDataset("test", args.data_dir)
    # test_set = preprocess_dataset(test_set, has_subject_idxs=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Valid set loaded with {len(test_set)} samples")
    print("Valid set sample:", test_set[0])
    print("num_classes", test_set.num_classes)
    print("seq_len", test_set.seq_len)
    print("num_channels", test_set.num_channels)

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)

            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)

            with torch.no_grad():
                y_pred = model(X)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(
            f"Epoch {epoch + 1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log(
                {"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss),
                 "val_acc": np.mean(val_acc)})

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(args.device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")
### models
class BasicConvClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,
            seq_len: int,
            in_channels: int,
            hid_dim: int = 128,
            num_heads: int = 8
            # hid_dim: int = 256
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            AttentionBlock(hid_dim, num_heads),
            ConvBlock(hid_dim, hid_dim),
            AttentionBlock(hid_dim, num_heads)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            kernel_size: int = 3,
            # p_drop: float = 0.2,
            p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        # self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.gelu(self.batchnorm2(X))

        return self.dropout(X)


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape from (batch_size, channels, seq_len) to (batch_size, seq_len, channels)
        x, _ = self.attention(x, x, x)
        x = self.norm(x)
        return x.permute(0, 2, 1)  # Change shape back to (batch_size, channels, seq_len)


### data import
class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path))

        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))

        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))

            return X, y, subject_idx
        else:
            return X, subject_idx

    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]

    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]

if __name__ == "__main__":
    run()