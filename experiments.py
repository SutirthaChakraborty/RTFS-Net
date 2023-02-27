import torch
from time import time
from tqdm import tqdm
from src.models import encoder

from src.models.layers.cnn_layers import ConvActNorm, FeedForwardNetwork, ConvolutionalRNN
from src.models.layers.rnn_layers import RNNProjection
from src.models.layers.attention import GlobalAttention

in_chan = 256
kernel_size = 5
dropout = 0.1
its = 100

model = ConvolutionalRNN(in_chan, in_chan * 2, kernel_size, dropout=dropout).cuda()
loss_fn = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model2 = GlobalAttention(in_chan, n_head=-1, kernel_size=kernel_size).cuda()
loss_fn2 = torch.nn.CrossEntropyLoss().cuda()
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)


model3 = GlobalAttention(in_chan, n_head=8, kernel_size=kernel_size).cuda()
loss_fn3 = torch.nn.CrossEntropyLoss().cuda()
optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.001, momentum=0.9)

model1_time = 0
model2_time = 0
model3_time = 0

pbar = tqdm(range(its))

for i in pbar:
    x = torch.rand((8, 256, 3200)).cuda().float()
    y = torch.rand((8, 256, 3200)).cuda().float()

    t1 = time()

    optimizer.zero_grad()
    outputs = model(x)
    loss1 = loss_fn(outputs, y)
    loss1.backward()
    optimizer.step()

    t2 = time()
    if i > 5:
        model1_time += t2 - t1

    # optimizer2.zero_grad()
    # outputs2 = model2(x)
    # loss2 = loss_fn2(outputs2, y)
    # loss2.backward()
    # optimizer2.step()

    t3 = time()
    if i > 5:
        model2_time += t3 - t2

    optimizer3.zero_grad()
    outputs3 = model3(x)
    loss3 = loss_fn3(outputs3, y)
    loss3.backward()
    optimizer3.step()

    t4 = time()
    if i > 5:
        model3_time += t4 - t3

    pbar.set_description(
        "Convolutional RNN time: {:.4f}, FFN time: {:.4f}, Global Attention time: {:.4f}".format(model1_time, model2_time, model3_time)
    )

model1_time /= its
model2_time /= its
model3_time /= its

print("Convolutional RNN time: {:.8f}".format(model1_time))
print("FNN time: {:.8f}".format(model2_time))
print("Global Attention time: {:.8f}".format(model3_time))
