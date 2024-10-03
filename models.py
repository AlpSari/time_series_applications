import torch

import torch.nn as nn


class Network(nn.Module):
    """MLP with series of Forward-BatchNorm-Activation layers."""

    def __init__(self, n_nodes_list=[1, 32, 1], activation_fn=torch.relu):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(node_in, node_out)
                for node_in, node_out in zip(n_nodes_list[:-2], n_nodes_list[1:-1])
            ]
        )
        self.batchnorm_layers = nn.ModuleList(
            [nn.BatchNorm1d(node_in) for node_in in n_nodes_list[1:-1]]
        )
        self.lastlayer = nn.Linear(n_nodes_list[-2], n_nodes_list[-1])
        self.activation_fn = activation_fn

    def forward(self, x):
        for layer_idx, (layer, bn) in enumerate(
            zip(self.layers, self.batchnorm_layers)
        ):
            x = self.activation_fn(bn(layer(x)))
        y = self.lastlayer(x)
        return y


class LSTMWithEncoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1, history_size=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # short hand for a 1 hidden layer NN
        net = lambda n_in, n_out: Network([n_in, 40, n_out])
        self.encoder = net(history_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=hidden_size, batch_first=True
        )
        self.h2o = net(hidden_size + self.input_size, self.output_size)  # i)

    def forward(self, inputs, hist):
        # inputs: [n_batch, n_sequence] (assumed 1 feature)
        # hist: [n_batch, n_history]
        h_0 = self.encoder(hist)[None, :, :]  # [1, n_batch, self.hidden_size]
        c_0 = torch.zeros(h_0.shape)  # [1, n_batch, self.hidden_size]

        # 1 layer LSTM with batch_first = True expects
        # INPUTS
        # input --> (n_batch, n_sequence, n_features)
        # h_0 --> (n_layer=1,n_batch, n_hidden)
        # c_0 --> (n_layer=1,n_batch, n_hidden)
        # OUTPUTS
        # hiddens: [n_batch, n_sequence, self.hidden_size]
        # Each hidden state is a vector of size self.hidden_size, representing the learned representation at each time step (n_sequence).
        # h_n: [self.output_size, n_features, self.hidden_size]
        # combined creates a feature vector which is the combination of the input + learned representation output from LSTM
        if inputs.ndim == 2:
            hiddens, (h_n, c_n) = self.lstm(inputs[:, :, None], (h_0, c_0))
            combined = torch.cat(
                (hiddens, inputs[:, :, None]), dim=2
            )  # [n_batch, n_sequence, n_hidden + n_features]
        else:
            hiddens, (h_n, c_n) = self.lstm(inputs, (h_0, c_0))
            combined = torch.cat(
                (hiddens), dim=2
            )  # [n_batch, n_sequence, n_hidden + n_features]

        # Since self.h2o predicts on the elements in last dimension, we can loop here by using `view`
        # For example, view [5,10,8] as [50,8] to get an output of shape [50, self.output_size], then reshape it back to [5, 10, self.output_size]
        h2o_input = combined.view(
            -1, self.hidden_size + self.input_size
        )  # [n_batch * n_sequence, n_hidden + n_features]
        y_predict = self.h2o(h2o_input).view(
            inputs.shape[0], inputs.shape[1]
        )  # [n_batch, n_sequence]
        return y_predict[:, -1][
            :, None
        ]  # Use the last output from the sequence as the output
