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
  # here, it is easy to run into numeric instability. So use stable version   #
  # of softmax. Don't forget the regularization!                              #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  for i in xrange(num_train):
    scores = X[i].dot(W)
 
    exp_scores = np.exp(scores)
    sumExp = np.sum(exp_scores)
    
    probExp = lambda k: np.exp(scores[k]) / sumExp
    
    loss += -np.log(probExp(y[i]))/num_train
    
    #regulization term
    regularizationTerm = reg*np.sum(np.square(W))
    loss += regularizationTerm
    
    #gradient
    num_classes = W.shape[1]
    for j in xrange(num_classes):
        if j==y[i]:
            dW[:,j] = (probExp(j)-1)*X[i]
        else:
            dW[:,j] = probExp(j)*X[i]
        
    dW += reg * W    

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  
  #scores -= np.max(scores, )

  exp_scores = np.exp(scores)
  sumExp = np.sum(exp_scores, axis=1, keepdims = True)
  probExp = exp_scores / sumExp

  loss = np.sum(-np.log(probExp[np.arange(num_train),y]))
  loss /= num_train
  #regulization term
  regularizationTerm = reg*np.sum(np.square(W))
  loss += regularizationTerm
  
  probExp[np.arange(num_train),y] -= 1
  dW = X.T.dot(probExp)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

