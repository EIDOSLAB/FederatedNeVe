# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""
import io
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import wandb
from flwr.common import FitIns
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    NDArrays, MetricsAggregationFn,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from my_federated.datasets.dataloader.loader import get_dataset_fed_path, load_loader, load_partition, load_transform
from my_federated.models import get_model
from my_federated.my_flwr.strategies.strategy_data import StrategyData
from my_federated.utils.trainer import eval_model


# pylint: disable=line-too-long

class FedAvgWConfig(FedAvg):
    """Federated Averaging strategy.
    Modified for some optimizations.

    Implementation based on https://arxiv.org/abs/1602.05629

    """

    def __init__(
            self,
            *,
            strategy_data: StrategyData = None,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            inplace: bool = True,
    ) -> None:
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
                         min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients,
                         min_available_clients=min_available_clients, evaluate_fn=evaluate_fn,
                         on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn,
                         accept_failures=accept_failures, initial_parameters=initial_parameters,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace, )
        self.strategy_data = strategy_data
        self.model, _ = get_model(strategy_data.dataset, strategy_data.model_name, strategy_data.device,
                                  False, strategy_data.use_groupnorm,
                                  strategy_data.groupnorm_channels)
        dataset_fed_path = get_dataset_fed_path(strategy_data.dataset_path, strategy_data.dataset,
                                                medmnist_size=strategy_data.medmnist_size,
                                                val_size=strategy_data.val_percentage / 100, seed=strategy_data.seed,
                                                num_clients=strategy_data.num_clients,
                                                concentration=strategy_data.concentration,
                                                ds_iid=strategy_data.dataset_iid, strategy_name=strategy_data.strategy)
        self.test_loader = load_partition(dataset_fed_path, -1, partitions=["test"],
                                          transform_2_pil=True)["test"]
        self.test_loader.transform = load_transform(strategy_data.dataset, strategy_data.model_name)["test"]
        self.test_loader = load_loader(self.test_loader, batch_size=32, shuffle=False)
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_test_accuracy = 0.0
        self._current_test_accuracy = 0.0

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        client_configs = super().configure_fit(server_round, parameters, client_manager)
        return client_configs

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        self.update_model_parameters(parameters_aggregated)
        self.test_aggregated_model(server_round)

        return parameters_aggregated, metrics_aggregated

    def update_model_parameters(self, parameters_aggregated):
        params_dict = list(zip(self.model.state_dict().keys(), parameters_aggregated.tensors))
        state_dict = OrderedDict({k: torch.tensor(np.load(io.BytesIO(v))) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def test_aggregated_model(self, server_round: int):
        results = eval_model(self.model, self.test_loader, self.strategy_data.dataset_task, self.strategy_data.device,
                             self.strategy_data.amp, server_round, "Test")
        print("---")
        print("Test results:", results)
        print("---")
        self._current_test_accuracy = results.get("accuracy", {}).get("top1", -1.0)
        wandb.log({"test": results}, commit=False)

    def aggregate_evaluate(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, EvaluateRes]],
            failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        if loss_aggregated is not None and loss_aggregated < self.best_val_loss:
            self.best_val_loss = loss_aggregated
            self.best_epoch = server_round - 1  # Wandb logs starts from 0 while server_round starts from 1
            self.best_test_accuracy = self._current_test_accuracy
        wandb.log({
            "best": {
                "val_loss": self.best_val_loss,
                "epoch": self.best_epoch,
                "test_accuracy": self.best_test_accuracy
            },
            "epoch": server_round - 1
        })
        return loss_aggregated, metrics_aggregated


def custom_on_fit_config_fn(server_round: int) -> dict:
    return {"round": server_round}
