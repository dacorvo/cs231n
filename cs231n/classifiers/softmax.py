import numpy as np
from random import shuffle

def softmax(x):
    shiftx = x - np.max(x)
    return np.exp(shiftx)/np.sum(np.exp(shiftx))

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(0, num_train):
      # Evaluate scores
      scores = X[i].dot(W)
      # Shift score values to prevent number overflow errors
      shifted_scores = scores - np.max(scores)
      # The sum of the scores exp is used both in loss and gradient
      sum_e_scores = 0
      for j in range(0, num_classes):
        sum_e_scores += np.exp(shifted_scores[j])
      # Loss is increased by the log of the sum minus the correct class score
      loss += np.log(sum_e_scores) - shifted_scores[y[i]]
      # Explaination for the calculus of the gradient
      #     L = log(sum(exp(s))) - s[y[i]]
      # Derivative of neperian log; ln(f)' = f'/f
      # d[j]L = d[j]sum(exp(s))/sum(exp(s)) - d[j]s[y[i]]
      # Derivative of sum: (f + g)' = f' + g'
      # d[j]L = sum(d[j]exp(s))/sum(exp(s)) - d[j]s[y[i]]
      # Derivative of exp: exp(f)' = f'.exp(f)
      # d[j]L = sum(d[j]s.exp(s))/sum(exp(s) - d[j]s[y[i]]
      # With, for a given class c: s[c] = w[c].X[i]
      #   d[j]s[c] = 0 if j <> c
      #   d[c]s[c] = X[i]
      # Then, in the first term of the gradient, all elements of the sum are
      # null BUT the one where j = k, and the second term of the gradient is
      # null except it j = y[i]
      #    d[j]L = X[i].exp(s[j])/sum(exp(s)) if j <> y[i]
      # d[y[i]]L = X[i].exp(s[j])/sum(exp(s)) -X[i]
      # Calculate gradient common part for each class
      for j in range(0, num_classes):
          dW[:,j] += X[i] * np.exp(shifted_scores[j])/sum_e_scores
      # For the correct class, the gradient has an extra term
      dW[:,y[i]] -= X[i]

  # We want mean values
  loss /= num_train
  dW /= num_train

  # Add regression
  loss += 0.5 * reg * np.sum(W*W)
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
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

