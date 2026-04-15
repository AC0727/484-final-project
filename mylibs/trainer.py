
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        optimizer,
        criterion,
        epochs,
        batch_size=32,
        device=None,
        scheduler=None,
        checkpoint_dir="checkpoints",
        checkpoint_freq=1,
        num_workers=4,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.criterion = criterion # ie the loss function
        self.epochs = epochs
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.model.to(self.device)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = float("inf")

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")

            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate()

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if self.scheduler:
                self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)

            # Periodic checkpoint
            if epoch % self.checkpoint_freq == 0:
                self._save_checkpoint(epoch, is_best=False)

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(self.train_loader, leave=False)

        for images, labels, _ in loop: # for each batch
            images = images.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1) # might have to change this depending on how the dataset is formatted

            self.optimizer.zero_grad() # zero the gradients

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward() 
            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5 # might have to change this depending on criterion?
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item()) # Update progress bar

        avg_loss = total_loss / len(self.train_loader)
        acc = correct / total

        return avg_loss, acc

    def _validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad(): # don't calculate the gradients
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1) # again, might have to change this

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5 # again, might need to change
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        acc = correct / total

        return avg_loss, acc

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if is_best:
            path = os.path.join(self.checkpoint_dir, "best_model.pth")
        else:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")


