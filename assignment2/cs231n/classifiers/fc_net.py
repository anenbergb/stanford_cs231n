from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        prev_dim = input_dim
        for i, output_dim in enumerate(hidden_dims + [num_classes]):
            self.params[f"W{i+1}"] = np.random.normal(loc=0.0,scale=weight_scale, size=(prev_dim, output_dim))
            self.params[f"b{i+1}"] = np.zeros((output_dim,), dtype=dtype)
            if normalization and i < self.num_layers - 1: 
                # batchnorm is only applied after hidden fully connected layers
                # gamma initialized to 1 keeps initial scaling neutral.
                self.params[f"gamma{i+1}"] = np.ones((output_dim,), dtype=dtype)
                # beta initialized to 0, meaning normalized activations have 0 mean
                self.params[f"beta{i+1}"] = np.zeros((output_dim,), dtype=dtype)
            prev_dim = output_dim

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

 
         ############################################################################
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        ############################################################################
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        
        layer_caches = []
        x = X
        for i in range(self.num_layers):
            x, fc_cache = affine_forward(x, self.params[f"W{i+1}"], self.params[f"b{i+1}"])
            if i == self.num_layers - 1:
                layer_caches.append(fc_cache)
            else:
                bn_cache = None
                drop_cache = None
                if self.normalization == "batchnorm":
                    x, bn_cache = batchnorm_forward(x, self.params[f"gamma{i+1}"], self.params[f"beta{i+1}"], self.bn_params[i])
                elif self.normalization == "layernorm":
                    x, bn_cache = layernorm_forward(x, self.params[f"gamma{i+1}"], self.params[f"beta{i+1}"], self.bn_params[i])
                x, relu_cache = relu_forward(x)
                if self.use_dropout:
                    x, drop_cache = dropout_forward(x, self.dropout_param)
                layer_caches.append((fc_cache, bn_cache, relu_cache, drop_cache))
        
        # If test mode return early.
        if y is None or mode == "test":
            return x

        loss, d_softmax = softmax_loss(x, y)
        for name, value in self.params.items():
           if name[0] == "W":
              loss += 0.5 * self.reg * np.sum(value * value)

        grads = {}
        d_output = d_softmax
        for i in range(self.num_layers - 1, -1, -1):
            if i == self.num_layers - 1:
                fc_cache = layer_caches[i]
            else:
                fc_cache, bn_cache, relu_cache, drop_cache = layer_caches[i]
                if drop_cache and self.use_dropout:
                    d_output = dropout_backward(d_output, drop_cache)
                d_output = relu_backward(d_output, relu_cache)
                if bn_cache and self.normalization == "batchnorm":
                    d_output, grads[f"gamma{i+1}"], grads[f"beta{i+1}"] = batchnorm_backward(d_output, bn_cache)
                elif bn_cache and self.normalization == "layernorm":
                    d_output, grads[f"gamma{i+1}"], grads[f"beta{i+1}"] = layernorm_backward(d_output, bn_cache)

            d_output, grads[f"W{i+1}"], grads[f"b{i+1}"] = affine_backward(d_output, fc_cache)
            grads[f"W{i+1}"] += self.reg * self.params[f"W{i+1}"]
            # batch_norm gamma and beta are not regularized

        return loss, grads
