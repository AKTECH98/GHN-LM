# Copyright (c) 2023. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Predicts parameters of a PyTorch model using one of the pretrained GHNs.

Example (use --debug 1 to perform additional sanity checks and print more information):
    export PYTHONPATH=$PYTHONPATH:./;  # to import ghn3 modules from the subfolder
    python examples/ghn_single_model.py --ckpt ghn3tm8.pt --arch resnet50

"""


import torch
import torchvision
from ppuda.config import init_config
from CustomGHN3 import from_pretrained, norm_check, Graph, Logger
from models import mini_gpt, GPTDecoderLM

# 1. Predict parameters of a PyTorch model using one of the pretrained GHNs

ghn = from_pretrained("ghn3tm8.pt").to("cpu")  # get a pretrained GHN

model = GPTDecoderLM(mini_gpt.GPTConfig(vocab_size=50257, max_seq_len=1024, n_layer=12, d_model=768, n_head=12, d_ff=3072, p_drop=0.1))
model = ghn(model)  # predict parameters of the model

graph = Graph(model)  # create a graph of the model once so that it can be reused for all training iterations
model.train()
ghn.train()
ghn.debug_level = 0  # disable debug checks and prints for every forward pass to ghn
opt = torch.optim.SGD(ghn.parameters(), lr=0.1)
logger = Logger(10)
for batch in range(10):
    opt.zero_grad()
    model = ghn(model, graph, keep_grads=True)  # keep gradients of predicted params to backprop to the ghn
    logits, _ = model(torch.randint(0, 50257, (2, 512)).to("cpu"))  # get logits
    loss = logits.abs().mean()  # some dummy loss
    loss.backward()
    total_ghn_norm = torch.nn.utils.clip_grad_norm_(ghn.parameters(), 5)
    opt.step()
    logger(batch, {'loss': loss.item(), 'grad norm': total_ghn_norm.item()})
