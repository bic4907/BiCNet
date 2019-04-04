import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, num_agents, is_cuda=True):
        super(Actor, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        # input (batch, s_dim) output (batch, 300)
        self.prev_dense = DenseNet(s_dim, 200, 200, output_activation=None, norm_in=True)
        # input (num_agents, batch, 200) output (num_agents, batch, num_agents * 2)
        self.comm_net = LSTMNet(num_agents, num_agents, num_layers=1)
        # input (batch, 2) output (batch, a_dim)
        self.post_dense = DenseNet(2, 32, a_dim)

    def forward(self, x):
        x = self.prev_dense(x)
        x = x.transpose(0, 1)
        x = self.comm_net(x)
        x = x.transpose(1, 0) # (batch, num_agnets, num_agents * 2)
        x = x.reshape(2)
        x = x.post_dense(x)
        return x


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, num_agent):
        super(Critic, self).__init__()

        self.num_agent = num_agent
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.norm1 = nn.BatchNorm1d(s_dim * num_agent)
        self.fc1 = nn.Linear(s_dim * num_agent, 64)
        self.fc2 = nn.Linear(64 + self.a_dim * num_agent, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

        self.norm1.weight.data.fill_(1)
        self.norm1.bias.data.fill_(0)

        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, x_n, a_n):
        x = x_n.reshape(-1, self.s_dim * self.num_agent)
        a = a_n.reshape(-1, self.a_dim * self.num_agent)
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.cat([x, a], 1)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



class DenseNet(nn.Module):
    def __init__(self, s_dim, hidden_dim, a_dim, norm_in=False, hidden_activation=nn.ReLU, output_activation=None):
        super(DenseNet, self).__init__()

        self._norm_in = norm_in

        if self._norm_in:
            self.norm1 = nn.BatchNorm1d(s_dim)

        self.dense1 = nn.Linear(s_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.dense4 = nn.Linear(hidden_dim, a_dim)

        if hidden_activation:
            self.hidden_activation = hidden_activation()
        else:
            self.hidden_activation = lambda x : x

        if output_activation:
            self.output_activation = output_activation()
        else:
            self.output_activation = lambda x : x

    def forward(self, x):
        if self._norm_in:
            x = self.norm1(x)
        x = self.hidden_activation(self.dense1(x))
        x = self.hidden_activation(self.dense2(x))
        x = self.hidden_activation(self.dense3(x))
        x = self.output_activation(self.dense4(x))
        return x


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1,
                 bias=True,
                 batch_fisrt=True,
                 dropout=False,
                 bidirectional=True):

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_fisrt,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, input, wh, wc):
        output, (hidden, cell) = self.lstm(input, (wh, wc))

        return output
