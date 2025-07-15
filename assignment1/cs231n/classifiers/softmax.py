from builtins import range
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W) # (1,C)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        
        # Incorrect first attempt
        # idx = y[i]
        # d_neg_logp = - 1 / p[idx]
        # for j in range(num_classes):
        #     delta_ij = 1 if j == idx else 0
        #     dW_idx = d_neg_logp * p[j] * (delta_ij - p[idx]) * X[i]
        #     dW[:,idx] += dW_idx

        for j in range(num_classes):
            delta_ij = j == y[i]
            dW[:, j] += - (delta_ij - p[j]) * X[i]

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    dW = dW / num_train + 2 * reg * W
    return loss, dW
    

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = np.matmul(X, W) # (N, C)
    scores -= np.max(scores, axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= np.sum(p, axis=-1, keepdims=True)
    logp = np.log(p)
    logp_y = logp[np.arange(num_train), y]
    loss = - np.sum(logp_y) / num_train + reg * np.sum(W * W)

    d_softmax = p.copy() # (N,C)
    d_softmax[np.arange(num_train), y] -= 1
    dW = np.matmul(X.T, d_softmax)
    dW = dW / num_train + 2 * reg * W

    return loss, dW
