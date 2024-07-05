import matplotlib.pyplot as plt
import numpy as np
import wandb
from flwr.server.client_proxy import ClientProxy


class ClientSamplerLogger:

    def __init__(self):
        self._history_active_clients: list[list[int]] = []
        self._current_max_clients: int = 0

    def update_sampled_clients(self, sampled_clients: list[ClientProxy], client_mapping: dict[str, int]):
        # Get the maximum client_id we reached (we have 1 client per client_id)
        num_clients = max([cidx for cid, cidx in client_mapping.items()]) + 1  # Clients ids starts from 0, so we add 1
        # Create a list containing for each client, (1) if it was sampled, (0) if it wasn't
        active_clients = [0 for _ in range(num_clients)]
        # Set to 1 all the sampled clients
        for sampled_client in sampled_clients:
            cidx = client_mapping[sampled_client.cid]
            active_clients[cidx] = 1

        # Update current max clients
        if len(active_clients) > self._current_max_clients:
            self._current_max_clients = len(active_clients)
            # Update old history logs with the new clients
            for history_page in self._history_active_clients:
                while len(history_page) < self._current_max_clients:
                    history_page.append(0)  # We set 0 since these clients were never been sampled

        # Add new data to history
        self._history_active_clients.append(active_clients)

    def log_sampled_fit_clients(self):
        # Log to wandb
        plot, plot_data = self._make_active_clients_plot()
        wandb.log({
            "selected_clients": plot,
            "selected_clients_data": plot_data
        }, commit=False)

    def _make_active_clients_plot(self):
        history_active_clients_data = np.array(self._history_active_clients).transpose()

        # Init the image with a dynamic size depending on the number of epochs
        fig, ax = plt.subplots(figsize=(2 + history_active_clients_data.shape[1] * 0.5, 2 * self._current_max_clients))
        # Add history data
        cax = ax.matshow(history_active_clients_data, cmap='Blues', vmin=0, vmax=1)

        # Add a colorbar with a static shrink
        colorbar = fig.colorbar(cax, orientation='vertical', shrink=0.5)
        colorbar.ax.tick_params(labelsize=16)

        # Update plot stats
        ax.set_xlabel('Epochs', fontsize=20)
        ax.set_ylabel('Clients', fontsize=20)
        ax.set_yticks(np.arange(self._current_max_clients))
        ax.set_yticklabels([f'{i}' for i in range(self._current_max_clients)], fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        xticks = np.linspace(0, history_active_clients_data.shape[1] - 1, num=10, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

        # Produce a wandb image
        wandb_plot = wandb.Image(fig)

        # Close the plt figure to save memory
        plt.close(fig)

        # Now also prepare the plot tabular data
        # Crea una tabella
        columns = [f"e_{i}" for i in range(history_active_clients_data.shape[1])]
        wandb_plot_data = wandb.Table(columns=columns)

        # Aggiungi righe alla tabella
        for row in history_active_clients_data:
            wandb_plot_data.add_data(*row)

        return wandb_plot, wandb_plot_data
