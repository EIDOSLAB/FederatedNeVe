import os

import flwr as fl
import numpy as np
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
        accuracy_1, accuracy_5 = training_stats["accuracy"]["top1"], training_stats["accuracy"]["top5"]

        # Return stats in a structured way
        results_data = {
            "loss": float(loss),
            "accuracy_top1": float(accuracy_1),
            "accuracy_top5": float(accuracy_5),
            "client_id": self.client_id,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        # In the first epoch also log the labels distribution
        if self.epoch < 1 and training_stats["labels_counter"]:
            label_counter = training_stats["labels_counter"]
            # Ordina le etichette per gestire etichette arbitrarie
            sorted_labels = sorted(label_counter.keys())
            label_distribution = [label_counter[label] for label in sorted_labels]
            # Normalizziamo i valori di distribuzione fra 0 e 100
            # Calcola la somma dei conteggi delle etichette
            total_count = sum(label_distribution)
            # Normalizza i conteggi in modo che la somma sia 100
            normalized_distribution = [(count / total_count) * 100 for count in label_distribution]
            # Add info to results
            results_data["data_distribution_count"] = np.array(normalized_distribution, dtype=float).tobytes()
            results_data["data_distribution_labels"] = np.array([int(slabel) for slabel in sorted_labels]).tobytes()
        return self.get_parameters(config={}), len(self.train_loader), results_data

    def _evaluate_method(self, parameters, config) -> tuple[float, int, dict]:
        # Validate the model on the validation-set
        stats = run(self.model, self.valid_loader, None, self.scaler, self.device,
                    self.amp, epoch=self.epoch, run_type="Validation")
        val_loss = stats["loss"]
        val_acc_1, val_acc_5 = stats["accuracy"]["top1"], stats["accuracy"]["top5"]

        # Validate the model on the test-set
        stats = run(self.model, self.test_loader, None, self.scaler, self.device,
                    self.amp, epoch=self.epoch, run_type="Test")
        test_loss = stats["loss"]
        test_acc_1, test_acc_5 = stats["accuracy"]["top1"], stats["accuracy"]["top5"]

        # Return stats in a structured way
        results_data = {
            # Validation
            "val_loss": float(val_loss),
            "val_accuracy_top1": float(val_acc_1),
            "val_accuracy_top5": float(val_acc_5),
            "val_size": len(self.valid_loader),
            # Test
            "test_loss": float(test_loss),
            "test_accuracy_top1": float(test_acc_1),
            "test_accuracy_top5": float(test_acc_5),
            "test_size": len(self.test_loader),
            # Other info
            "client_id": self.client_id,
            "confusion_matrix": stats["confusion_matrix"].tobytes(),
            "confusion_matrix_shape_d0": stats["confusion_matrix"].shape[0],
            "confusion_matrix_shape_d1": stats["confusion_matrix"].shape[1],
        }
        return float(val_loss), len(self.valid_loader), results_data

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
