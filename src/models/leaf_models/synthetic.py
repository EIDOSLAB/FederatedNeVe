import torch
from torch import nn


class LeafSyntheticModel(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(LeafSyntheticModel, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        # Definire un livello denso (equivalente a tf.layers.dense)
        self.dense = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x):
        # Applicare il livello denso e la funzione di attivazione sigmoid
        logits = torch.sigmoid(self.dense(x))
        return logits
