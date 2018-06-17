# Delta Orthogonal Initialization for Lua Torch


An implementation of the weight initialization proposed in the paper ["Dynamical Isometry and a Mean Field Theory of CNNs:
How to Train 10,000-Layer Vanilla Convolutional Neural Networks"](https://arxiv.org/abs/1806.05393) by Xiao et al.

To initialize the weights in a single layer, call `makeDeltaOrthogonal()`:

    local conv = nn.SpatialConvolution(in, out, 3, 3)
    makeDeltaOrthogonal(conv.weight)
    
To initialize all layers in a network, call `initAll()`

    local model = nn.Sequential(...)
    initAll(model)
    
License: MIT
