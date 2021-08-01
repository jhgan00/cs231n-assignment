from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = len(y)
    
    for i in range(num_train):
        
        label = y[i]
        
        x = X[i]
        z = x.dot(W)
        z -= z.max()
        
        numerator = np.exp(z)
        denominator = numerator.sum()
        softmax = numerator / denominator
        
        yhat = softmax[label]
        loss -= np.log(yhat)
        
        dw = softmax.copy()
        dw[label] = yhat - 1
        dW += x[..., np.newaxis] * dw[np.newaxis, ...]
        
        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    loss /= num_train
    dW /= num_train
    
    loss += reg * (W * W).sum()
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)  # (d, c)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    
    Z = X.dot(W)
    Z -= Z.max(axis=1, keepdims=True)
    
    numerator = np.exp(Z)
    denominator = np.exp(Z).sum(axis=1, keepdims=True)
    
    softmax = numerator / denominator
    yhat = softmax[np.arange(num_train), y]

    loss -= np.log(yhat).sum() / num_train
    
    dw = softmax.copy()  # (n, c)
    dw[np.arange(num_train), y] = yhat - 1
    dW = (X[..., np.newaxis] * dw[:, np.newaxis, :]).sum(axis=0) / num_train  # (n, d)
    dW += 2 * reg * W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
