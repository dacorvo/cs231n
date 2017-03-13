import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        # This is the wrong class for this data, but it still receives a
        # good score relative to the correct class
        # This contributes to the loss
        loss += margin
        # The gradient for this class is increased by the value of the
        # data in all dimensions for this datapoint
        dW[:,j] += X[i]
        # This also contributes to decrease the gradient of the correct class
        # in reverse proportions
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Average gradient as well
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Also add regularisation to the gradient
  # regularization matrix is 0.5 * reg * W*W, so dReg = 0.5 * reg * 2 * W
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  # Compute scores with a simple matrix multiplication
  scores = X.dot(W)
  # Extract correct scores using numpy indexing capabilities
  # First vector imdicates the index for the first dimension (a range)
  # second vector is the correct class index y
  correct_scores = scores[np.arange(0,num_train),y]
  # Calculate margin for each individual score. Note that we need to transpose
  # the scores matrix to benefit from broadcast
  margin = scores.T - correct_scores + 1
  # Force margin to zero for correct classes (using y as indices)
  margin[y,np.arange(0,num_train)] = 0
  # We accumulate loss if margin is strictly positive
  loss = np.sum(margin[margin > 0])

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # We will reuse the loss matrix we calculated before
  # First, convert our margin matrix to a binary one
  dL = (margin > 0).astype(int)
  # We still use a transposed matrix (which is counter intuitive):
  # In each datapoint column, the row that corresponds to the 'correct' class
  # weights negatively as much as the sum of the 'wrong' classes weight
  dL[y,np.arange(0,num_train)] = -1 * np.sum(dL, axis=0)
  # The gradient is the result of the multiplication of the new matrix with the
  # data matrix.
  # Note that the result must be transposed back to match the gradient shape
  dW = (dL.dot(X)).T
  # Normalize the matrix
  dW /= num_train
  # Add regularization
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
