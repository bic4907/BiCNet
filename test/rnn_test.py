from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

input_size = 10
hidden_size = 30
output_size = 10
batch_size = 3
length = 4
num_layers = 3

rnn = nn.LSTM(input_size,hidden_size,num_layers=num_layers,bias=True,batch_first=True,bidirectional=True)

input = Variable(torch.randn(batch_size,length,input_size)) # B,T,D
print(input.size())
hidden = Variable(torch.zeros(num_layers*2,batch_size,hidden_size)) # (num_layers * num_directions, batch, hidden_size)
cell = Variable(torch.zeros(num_layers*2,batch_size,hidden_size)) # (num_layers * num_directions, batch, hidden_size)

output, (hidden,cell) = rnn(input,(hidden,cell))

print(output.size())
print(hidden.size())
print(cell.size())


linear = nn.Linear(hidden_size*2,output_size)
output = F.softmax(linear(output),1)
output.size()