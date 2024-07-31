import os

import flwr as fl
import torch

from src.my_flwr.clients.default import AFederatedClient
from src.utils.trainer import run


class FederatedDefaultClient(AFederatedClient, fl.client.NumPyClient):
    def _update_components(self):
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()

    def _fit_method(self, parameters, config) -> tuple[list, int, dict]:
        # Perform one epoch of training
        training_stats = run(self.model, self.train_loader, self.optimizer, self.scaler, self.device,
                             self.amp, epoch=self.epoch, run_type="Train")
        # Unwrap training stats
        loss = training_stats["loss"]
        accuracy_1 = training_stats["accuracy"]["top1"]

        # Return stats in a structured way
        results_data = {
            "loss": float(loss),
            "accuracy_top1": float(accuracy_1),
            "client_id": self.client_id,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        return self.get_parameters(config={}), len(self.train_loader.dataset), results_data

    def _evaluate_method(self, parameters, config) -> tuple[float, int, dict]:
        # Validate the model on the validation-set
        val_stats = run(self.model, self.valid_loader, None, self.scaler, self.device,
                        self.amp, epoch=self.epoch, run_type="Validation")
        val_loss = val_stats["loss"]
        val_acc_1 = val_stats["accuracy"]["top1"]

        # Validate the model on the test-set
        test_stats = run(self.model, self.test_loader, None, self.scaler, self.device,
                         self.amp, epoch=self.epoch, run_type="Test")
        test_loss = test_stats["loss"]
        test_acc_1 = test_stats["accuracy"]["top1"]

        # Return stats in a structured way
        results_data = {
            # Validation
            "val_loss": float(val_loss),
            "val_accuracy_top1": float(val_acc_1),
            "val_size": len(self.valid_loader.dataset),
            # Test
            "test_loss": float(test_loss),
            "test_accuracy_top1": float(test_acc_1),
            "test_size": len(self.test_loader.dataset),
            # Other info
            "client_id": self.client_id,
        }
        return float(val_loss), len(self.valid_loader.dataset), results_data

    def _load(self):
        if not os.path.exists(self.disk_folder):
            os.makedirs(self.disk_folder)

        if not os.path.exists(self.get_client_checkpoint_path()):
            return

        checkpoint = torch.load(self.get_client_checkpoint_path())
        self.epoch = checkpoint["epoch"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])

    def _save(self):
        torch.save(
            {
                "epoch": self.epoch,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
            },
            self.get_client_checkpoint_path()
        )
