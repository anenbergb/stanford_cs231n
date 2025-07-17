from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    N = x.shape[0]
    x_flat = x.reshape(N, -1) # (N,D)
    out = np.matmul(x_flat, w) + b # (N,M)
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N = x.shape[0]
    x_flat = x.reshape(N, -1) # (N,D)

    # (N,M) @ (M,D). the matmulis doing a sum across the M dimension
    dx = np.matmul(dout, w.T).reshape(x.shape)
    dw = np.matmul(x_flat.T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.where(x > 0, x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    dx = np.where(x > 0, dout, 0)
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    num_train, num_classes = x.shape
    # x is scores
    scores = x - np.max(x, axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= np.sum(p, axis=-1, keepdims=True)
    logp = np.log(p)
    logp_y = logp[np.arange(num_train), y]
    loss = - np.sum(logp_y) / num_train

    dx = p.copy() # (N,C)
    dx[np.arange(num_train), y] -= 1
    dx /= num_train
   
    return loss, dx



def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.
    https://arxiv.org/abs/1502.03167

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))


    sample_mean = np.mean(x, axis=0)  # (D,)
    sample_var = np.var(x, axis=0)

    out, cache = None, None
    if mode == "train":
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_norm + beta

        # running mean and var are only used at test time
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = {
            "x_norm": x_norm,
            "gamma": gamma,
            "x_norm_numerator": x - sample_mean,
            "x_norm_denominator": 1 / np.sqrt(sample_var + eps),
        }

    elif mode == "test":
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.
    https://arxiv.org/abs/1502.03167

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """    
    x_norm = cache["x_norm"] # (N,D)
    gamma = cache["gamma"]    
    x_norm_numerator = cache["x_norm_numerator"]
    x_norm_denominator = cache["x_norm_denominator"] # 1/denominator


    N = dout.shape[0]
    dbeta = np.sum(dout, axis=0) # (D,)
    dgamma = np.sum(dout * x_norm, axis=0) # (D,)

    d_xnorm = dout * gamma # (N,D) broadcast the gamma
    
    d_var = -0.5 * np.sum(d_xnorm * x_norm_numerator * np.pow(x_norm_denominator, 3), axis=0) # (D,)
    d_mean = - np.sum(d_xnorm * x_norm_denominator, axis=0) - (2/N) * d_var * np.sum(x_norm_numerator, axis=0)

    dx = d_xnorm * x_norm_denominator + (2/N) * d_var * x_norm_numerator + (1/N) * d_mean
    
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    eps = ln_param.get("eps", 1e-5)

    sample_mean = np.mean(x, axis=1, keepdims=True) # (N,1)
    sample_var = np.var(x, axis=1, keepdims=True)

    x_norm_numerator = x - sample_mean
    x_norm_denominator = 1 / np.sqrt(sample_var + eps)
    x_norm = x_norm_numerator * x_norm_denominator
    out = gamma * x_norm + beta

    cache = {
        "x_norm": x_norm,
        "gamma": gamma,
        "x_norm_numerator": x_norm_numerator,
        "x_norm_denominator": x_norm_denominator,
    }
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x_norm = cache["x_norm"] # (N,D)
    gamma = cache["gamma"]    
    x_norm_numerator = cache["x_norm_numerator"]
    x_norm_denominator = cache["x_norm_denominator"] # 1/denominator


    N, D = dout.shape
    dbeta = np.sum(dout, axis=0, ) # (D,)
    dgamma = np.sum(dout * x_norm, axis=0) # (D,)
    d_xnorm = dout * gamma # (N,D) broadcast the gamma
    
    d_var = -0.5 * np.sum(d_xnorm * x_norm_numerator * np.pow(x_norm_denominator, 3), axis=1, keepdims=True) # (N,)
    d_mean = - np.sum(d_xnorm * x_norm_denominator, axis=1, keepdims=True) - (2/D) * d_var * np.sum(x_norm_numerator, axis=1, keepdims=True)

    dx = d_xnorm * x_norm_denominator + (2/D) * d_var * x_norm_numerator + (1/D) * d_mean
    
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        # inverted dropout
        mask = (np.random.rand(*x.shape) < p) / p # probability of keeping a neuron
        out = x * mask
        
    elif mode == "test":
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    pad = conv_param.get("pad", 0)
    stride = conv_param.get("stride", 1)

    N,C,H,W = x.shape
    F,C,HH,WW = w.shape

    out_h = (H + 2 * pad - HH) // stride + 1
    out_w = (W + 2 * pad - WW) // stride + 1
    out = np.zeros((N,F,out_h, out_w), dtype=x.dtype)

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), constant_values=0.0)
    for i in range(0, H + 2 * pad - HH + 1,  stride):
        for j in range(0, W + 2 * pad - WW + 1, stride):
            x_patch = x_pad[...,i:i+HH,j:j+WW] # (N,C,HH,WW)
            # (N,1,C,HH,WW) * (1,F,C,HH,WW)
            prod = x_patch[:,None,...] * w[None,...]
            out[:,:, i // stride, j // stride] = np.sum(prod, axis=(2,3,4))# (N,F)            
    out += b.reshape(1,F,1,1)

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache

    pad = conv_param.get("pad", 0)
    stride = conv_param.get("stride", 1)

    N,C,H,W = x.shape
    F,C,HH,WW = w.shape

    out_h = (H + 2 * pad - HH) // stride + 1
    out_w = (W + 2 * pad - WW) // stride + 1

    No, Fo, Ho, Wo = dout.shape
    assert No == N
    assert Fo == F
    assert Ho == out_h
    assert Wo == out_w

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), constant_values=0.0)

    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)

    for i in range(0, H + 2 * pad - HH + 1,  stride):
        for j in range(0, W + 2 * pad - WW + 1, stride):
            x_patch = x_pad[...,i:i+HH,j:j+WW] # (N,C,HH,WW)
            
            # (N,F,1,1,1) * (1,F,C,HH,WW), sum along F to get (N,C,HH,WW)
            dout_patch = dout[:,:, i // stride, j // stride][...,None,None,None]
            dx_patch = np.sum(dout_patch * w[None,...], axis=1)
            dx_pad[...,i:i+HH,j:j+WW] += dx_patch
            
            # (N,F,1,1,1) * (N,1,C,HH,WW), sum along N to get (F,C,HH,WW)
            dw += np.sum(dout_patch * x_patch[:,None,...], axis=0)

    dx = dx_pad[..., pad:-pad, pad:-pad]
    db = np.sum(dout, axis=(0,2,3))

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    N,C,H,W = x.shape
    pheight = pool_param["pool_height"]
    pwidth = pool_param["pool_width"]
    stride = pool_param["stride"]

    Hout = (H - pheight) // stride + 1
    Wout = (W - pwidth) // stride + 1

    out = np.zeros((N,C,Hout,Wout), dtype = x.dtype)
    for i in range(0, H - pheight + 1, stride):
        for j in range(0, W - pwidth + 1, stride):
            x_patch = x[...,i:i+pheight,j:j+pwidth] # (N,C,pheight,pwidth)
            x_patch_max = np.max(x_patch, axis=(2,3)) # (N,C)
            # (N,C,1,1)
            out[..., i // stride, j // stride] = x_patch_max

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x

    Rather than constructing the mask with each iteration (e.g. x_patch_flat_mask)
    and then reshaping. We could instead try to directly index the dout_patch
    and set those values in the dx matrix.
    This might be possible with clever indexing and using 
    np.divmod(x_patch_flat_max_idx, pwidth)
    """
    x, pool_param = cache

    N,C,H,W = x.shape
    pheight = pool_param["pool_height"]
    pwidth = pool_param["pool_width"]
    stride = pool_param["stride"]

    Hout = (H - pheight) // stride + 1
    Wout = (W - pwidth) // stride + 1

    No, Co, Ho, Wo = dout.shape
    assert No == N
    assert Co == C
    assert Ho == Hout
    assert Wo == Wout

    dx = np.zeros_like(x)

    n_idx = np.arange(N)[:, None] # (N,1)
    c_idx = np.arange(C)[None, :] # (1, C)
    for i in range(0, H - pheight + 1, stride):
        for j in range(0, W - pwidth + 1, stride):
            x_patch = x[...,i:i+pheight,j:j+pwidth] # (N,C,pheight,pwidth)
            x_patch_flat = x_patch.reshape((N,C,-1)) # (N,C,pheight*pwidth)
            x_patch_flat_max_idx = np.argmax(x_patch_flat, axis=2) # (N,C)

            x_patch_flat_mask = np.zeros_like(x_patch_flat)
            # broadcastable indexing to set 1 in each (N,C,:) slice
            x_patch_flat_mask[n_idx, c_idx, x_patch_flat_max_idx] = 1
            x_patch_mask = x_patch_flat_mask.reshape((N,C,pheight, pwidth))

            # (N,C,1,1)
            dout_patch = dout[..., i // stride, j // stride][..., None, None]            
            dx[...,i:i+pheight,j:j+pwidth] += x_patch_mask * dout_patch

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N,C,H,W = x.shape
    xtrans = x.transpose((0,2,3,1)) # (N,H,W,C)
    xflat = xtrans.reshape((-1, C)) # (N*H*W,C)
    out_flat, cache = batchnorm_forward(xflat, gamma, beta, bn_param)
    out_trans = out_flat.reshape((N,H,W,C))
    out = out_trans.transpose((0,3,1,2))
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N,C,H,W = dout.shape
    dout_flat = dout.transpose((0,2,3,1)).reshape((-1,C))
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    dx = dx_flat.reshape((N,H,W,C)).transpose((0,3,1,2))
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    
    Very similar to layer norm, but we're splitting the channels into groups
    """
    eps = gn_param.get("eps", 1e-5)

    N,C,H,W = x.shape
    # layer norm would average across C,H,W (N,C*H*W)
    # group norm is (N,G,(C//G)*H*W) 
    x_group = x.reshape((N, G, C // G, H, W))
    group_mean = np.mean(x_group, axis=(2,3,4), keepdims=True) # (N,G,1,1,1)
    group_var = np.var(x_group, axis=(2,3,4), keepdims=True) # (N,G,1,1,1)

    x_group_norm_numerator = x_group - group_mean
    x_group_norm_denominator = 1 / np.sqrt(group_var + eps)
    x_group_norm = x_group_norm_numerator * x_group_norm_denominator # (N,G, C // G, H, W)
    x_norm = x_group_norm.reshape((N,C,H,W))
    out = gamma * x_norm + beta

    cache = {
        "G": G,
        "x_norm": x_norm,
        "gamma": gamma,
        "x_group_norm_numerator": x_group_norm_numerator,
        "x_group_norm_denominator": x_group_norm_denominator,
    }
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    x_norm = cache["x_norm"] # (N,C,H,W)
    gamma = cache["gamma"]
    G = cache["G"]

    x_group_norm_numerator = cache["x_group_norm_numerator"] # (N,G,C//G,H,W)
    # 1/denominator (N,G,1,1,1)
    x_group_norm_denominator = cache["x_group_norm_denominator"]


    N, C, H , W = dout.shape

    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True) # (1,C,1,1)
    dgamma = np.sum(dout * x_norm, axis=(0,2,3), keepdims=True) # (1,C,1,1)
    dx_norm = dout * gamma # (N,C,H,W)
    dx_group_norm = dx_norm.reshape((N,G, C // G, H, W))


    d_var = -0.5 * np.sum(dx_group_norm * x_group_norm_numerator * np.pow(x_group_norm_denominator, 3), axis=(2,3,4), keepdims=True) # (N,G,1,1,1)
    
    num = np.prod((C//G, H, W))
    d_mean = - np.sum(dx_group_norm * x_group_norm_denominator, axis=(2,3,4), keepdims=True) - (2/num) * d_var * np.sum(x_group_norm_numerator, axis=(2,3,4), keepdims=True)

    # (N,G,C//G,H,W)
    dx_group = dx_group_norm * x_group_norm_denominator + (2/num) * d_var * x_group_norm_numerator + (1/num) * d_mean
    dx = dx_group.reshape((N,C,H,W))

    return dx, dgamma, dbeta
