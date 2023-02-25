import torch
from time import time
from tqdm import tqdm
from src.models import encoder

from src.models.layers.attention import FeedForwardNetwork
from src.models.layers.cnn_layers import ConvActNorm

in_chan = 256
kernel_size = 5
dropout = 0.1
its = 1000

model = FeedForwardNetwork(in_chan, in_chan * 2, kernel_size, dropout=dropout).cuda()

loss_fn = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model2 = ConvActNorm(in_chan, in_chan, kernel_size, padding=(kernel_size - 1) // 2, norm_type="gLN", act_type="ReLU").cuda()

loss_fn2 = torch.nn.CrossEntropyLoss().cuda()
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)

model1_time = 0
model2_time = 0

pbar = tqdm(range(its))

for i in pbar:
    x = torch.rand((8, 256, 3200)).cuda()
    y = torch.rand((8, 256, 3200)).cuda()

    t1 = time()

    optimizer.zero_grad()
    outputs = model(x)
    loss1 = loss_fn(outputs, y)
    loss1.backward()
    optimizer.step()

    t2 = time()
    model1_time += t2 - t1

    optimizer2.zero_grad()
    outputs2 = model2(x)
    loss2 = loss_fn2(outputs2, y)
    loss2.backward()
    optimizer2.step()

    t3 = time()
    model2_time += t3 - t2

    pbar.set_description("FFN time: {:.4f}, Conv time: {:.4f}".format(model1_time, model2_time))

model1_time /= its
model2_time /= its

print("FFN time: {:.8f}".format(model1_time))
print("Conv time: {:.8f}".format(model2_time))
