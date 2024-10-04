import torch
from torch import nn


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
        # h_0 --> (n_layer=1, n_batch, n_hidden)
        # c_0 --> (n_layer=1, n_batch, n_hidden)
        # OUTPUTS
        # hiddens: [n_batch, n_sequence, self.hidden_size]  Hidden states at each time step, representing the learned representation.
        # h_n: [1, n_batch, self.hidden_size]:  the final hidden state for each element in the sequence
        # combined: [n_batch, self.hidden_size + n_features ]:  a feature vector which is the combination of the input (last timestep) + learned representation output from LSTM

        if inputs.ndim == 2:
            hiddens, (h_n, c_n) = self.lstm(inputs[:, :, None], (h_0, c_0))
            combined = torch.cat(
                (hiddens[:, -1, :].squeeze(), inputs[:, -1, None]), dim=1
            )
        else:
            hiddens, (h_n, c_n) = self.lstm(inputs, (h_0, c_0))
            combined = torch.cat(
                (hiddens[:, -1, :].squeeze(), inputs[:, -1, :]), dim=1
            )  # [n_batch, n_hidden + n_features]

        # Note that h_n[-1,:, :] == hiddens[:,-1,:]
        y_predict = self.h2o(combined)  # [n_batch, self.output_size]
        return y_predict
