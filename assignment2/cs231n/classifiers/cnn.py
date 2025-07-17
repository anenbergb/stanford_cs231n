from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.params["W1"] = np.random.normal(loc=0.0, scale = weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
        self.params["b1"] = np.zeros((num_filters,), dtype=dtype)
        hidden_dim_flat = num_filters * input_dim[1] * input_dim[2] // 4
        self.params["W2"] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim_flat, hidden_dim))
        self.params["b2"] = np.zeros((hidden_dim,), dtype=dtype)
        self.params["W3"] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params["b3"] = np.zeros((num_classes,), dtype=dtype)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.

        conv - relu - 2x2 max pool - affine - relu - affine - softmax
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        x = X
        x, conv1_cache = conv_relu_pool_forward(x, W1, b1, conv_param, pool_param)
        conv1_output_shape = x.shape
        x = x.reshape((x.shape[0], -1))
        x, affine2_cache = affine_relu_forward(x, W2, b2)
        scores, affine3_cache = affine_forward(x, W3, b3)

        if y is None:
            return scores

        loss, d_softmax = softmax_loss(scores, y)
        for name, value in self.params.items():
           if name[0] == "W":
              loss += 0.5 * self.reg * np.sum(value * value)
        
        grads = {}
        dout, grads["W3"], grads["b3"] = affine_backward(d_softmax, affine3_cache)
        dout, grads["W2"], grads["b2"] = affine_relu_backward(dout, affine2_cache)
        dout = dout.reshape(conv1_output_shape)
        dout, grads["W1"], grads["b1"] = conv_relu_pool_backward(dout, conv1_cache)

        return loss, grads
