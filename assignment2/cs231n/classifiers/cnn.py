import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32,use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
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
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    F = filter_size
    self.params['W1'] = np.random.randn(num_filters, C, F, F) * weight_scale
    self.params['b1'] = np.zeros(num_filters)
    # Evaluate the dimension D of the flattened conv output
    # Let's assume the conv layer preserves the input, that is later divided by
    # 2 along each dimension by the pool layer
    D = int(num_filters * H * W /4)
    self.params['W2'] = np.random.randn(D, hidden_dim) * weight_scale
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b3'] = np.zeros(num_classes)
    self.use_batchnorm = use_batchnorm
    if self.use_batchnorm:
        self.params['gamma1'] = np.ones(num_filters)
        self.params['beta1'] = np.zeros(num_filters)
        self.params['gamma2'] = np.ones(hidden_dim)
        self.params['beta2'] = np.zeros(hidden_dim)
        self.bn_params = [{'mode': 'test' },{'mode': 'test'}]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if self.use_batchnorm:
        mode = 'test' if y is None else 'train'
        for bn_param in self.bn_params:
            bn_param['mode'] = mode
        conv, convcache = conv_bnorm_relu_pool_forward(X, W1, b1,
                                                       self.params["gamma1"],
                                                       self.params["beta1"],
                                                       conv_param,
                                                       pool_param,
                                                       self.bn_params[0])
        hidden, hcache = affine_bnorm_relu_forward(conv, W2, b2,
                                                  self.params["gamma2"],
                                                  self.params["beta2"],
                                                  self.bn_params[1])
    else:
        conv, convcache = conv_relu_pool_forward(X, W1, b1,
                                                 conv_param, pool_param)
        hidden, hcache = affine_relu_forward(conv, W2, b2)
    scores, fcache = affine_forward(hidden, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    # Add regularization
    loss += 0.5 * self.reg * np.sum(np.square(self.params['W1']))
    loss += 0.5 * self.reg * np.sum(np.square(self.params['W2']))
    loss += 0.5 * self.reg * np.sum(np.square(self.params['W3']))
    # Calculate third layer gradients
    dl3, grads['W3'], grads['b3'] = affine_backward(dout, fcache)
    # Calculate second layer gradients
    if self.use_batchnorm:
        dl2, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = \
                affine_bnorm_relu_backward(dl3, hcache)
        # Calculate conv layer gradients
        dX, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = \
                conv_bnorm_relu_pool_backward(dl2, convcache)
    else:
        dl2, grads['W2'], grads['b2'] = affine_relu_backward(dl3, hcache)
        # Calculate conv layer gradients
        dX, grads['W1'], grads['b1'] = conv_relu_pool_backward(dl2, convcache)
    # Add regularization for each gradient
    grads['W1'] += self.reg * self.params['W1']
    grads['W2'] += self.reg * self.params['W2']
    grads['W3'] += self.reg * self.params['W3']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
class ConvolutionalNet(object):
  """
  A convolutional neural network with an arbitrary number of convolutional
  layers and an arbitrary number of affine layers, ReLU nonlinearities, and a
  softmax loss function. It also implements batch normalization as an option.
  For a network with N conv layers and M affine layers the architecture is:
  
  {affine - [batch norm] - relu - pool} x (N) 
                 -> affine [batch norm] x (M)
                 -> softmax
  
  Similar to the ThreeLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def _get_wb_names(self, i):
        return 'W%d' % i, 'b%d' % i

  def _get_bn_names(self, i):
        return 'gamma%d' % i, 'beta%d' % i

  def __init__(self, conv_dims, affine_dims, input_dim=(3,32,32),
               num_classes=10, use_batchnorm=False, reg=0.0,
               weight_scale=1e-3, dtype=np.float32):
    """
    Initialize a new ConvolutionalNet.
    
    Inputs:
    - conv_dims: A list of triplets giving the size of each conv layer:
        num_filters, filter_size and pool_size
    - affine_dims: A list of integers giving the size of each affine layer.
    - input_dim: A C, H, W triplet giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    """
    self.use_batchnorm = use_batchnorm
    self.num_conv = len(conv_dims)
    self.num_affine = len(affine_dims)
    self.reg = reg
    self.dtype = dtype
    self.params = {}
    self.conv_params = []
    self.pool_params = []
    self.bn_params = []

    ############################################
    # Initialize the parameters of the network #
    ############################################
    C, H, W = input_dim

    # Initialize first the conv-relu-pool layers
    for i in range(1, self.num_conv + 1):
        wname, bname = self._get_wb_names(i)
        num_filters, filter_size, pool_size = conv_dims[i - 1]
        self.params[wname] = np.random.randn(num_filters,
                                             C,
                                             filter_size,
                                             filter_size) * weight_scale
        self.params[bname] = np.zeros(num_filters)
        conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}
        self.conv_params.append(conv_param)
        pool_param = {'pool_height': pool_size,
                      'pool_width': pool_size,
                      'stride': pool_size}
        self.pool_params.append(pool_param)
        if self.use_batchnorm:
            gammaname, betaname = self._get_bn_names(i)
            self.params[gammaname] = np.ones(num_filters)
            self.params[betaname] = np.zeros(num_filters)
            self.bn_params.append({'mode': 'test'})
        # The next conv layer will work on a reduced image
        H = int(H/pool_size)
        W = int(W/pool_size)
    # Evaluate the size of the flattened conv output
    D = int(num_filters*H*W)

    # Now initialize the affine layers
    for i in range (self.num_conv + 1, self.num_conv + self.num_affine + 1):
        wname, bname = self._get_wb_names(i)
        H = affine_dims[i - self.num_conv -1]
        self.params[wname] = np.random.randn(D, H) * weight_scale
        self.params[bname] = np.zeros(H)
        if self.use_batchnorm:
            gammaname, betaname = self._get_bn_names(i)
            self.params[gammaname] = np.ones(H)
            self.params[betaname] = np.zeros(H)
            self.bn_params.append({'mode': 'test'})
        # Next affine layer will have H as input dimension
        D = H

    # Final layer is also an affine layer, without batch norm
    wname, bname = self._get_wb_names(self.num_conv + self.num_affine + 1)
    self.params[wname] = np.random.randn(D, num_classes) * weight_scale
    self.params[bname] = np.zeros(num_classes)

    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Compute loss and gradient for the convolutional net.

    Input / output: Same as ThreeLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they behave differently
    # during training and testing.
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    scores = None
    ############################################################################
    # Implement the forward pass for the convolutional net, computing the
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    # Store intermediate results in the H variable
    H = X
    caches = []

    # First, evaluate forward pass for the conv-relu-pool layers
    for i in range(1, self.num_conv + 1):
        wname, bname = self._get_wb_names(i)
        W = self.params[wname]
        b = self.params[bname]
        if self.use_batchnorm:
            gammaname, betaname = self._get_bn_names(i)
            gamma = self.params[gammaname]
            beta = self.params[betaname]
            H, cache = conv_bnorm_relu_pool_forward(H, W, b,
                                                    gamma,beta,
                                                    self.conv_params[i-1],
                                                    self.pool_params[i-1],
                                                    self.bn_params[i-1])
        else:
            H, cache = conv_relu_pool_forward(H, W, b,
                                              self.conv_params[i-1],
                                              self.pool_params[i-1])
        caches.append(cache)

    # Second, evaluate forward pass for the intermediate affine layers
    for i in range (self.num_conv + 1, self.num_conv + self.num_affine + 1):
        wname, bname = self._get_wb_names(i)
        W = self.params[wname]
        b = self.params[bname]
        if self.use_batchnorm:
            gammaname, betaname = self._get_bn_names(i)
            gamma = self.params[gammaname]
            beta = self.params[betaname]
            H, cache = affine_bnorm_relu_forward(H, W, b,
                                                 gamma, beta,
                                                 self.bn_params[i-1])
        else:
            H, cache = affine_relu_forward(H, W, b)
        caches.append(cache)

    # Then, evaluate forward pass for the last affine layer (no ReLU, no batch
    # normalization)
    wname, bname = self._get_wb_names(self.num_conv + self.num_affine + 1)
    W = self.params[wname]
    b = self.params[bname]
    H, cache = affine_forward(H, W, b)
    caches.append(cache)
    # Finally, assign scores
    scores = H

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # Implement the backward pass for the fully-connected net, storing the     #
    # loss in the loss variable and gradients in the grads dictionary.         #
    # As a convention, the output variable grads[k] holds the gradients for    #
    # for self.params[k].                                                      #
    ############################################################################
    loss, dout = softmax_loss(scores, y)

    # First, perform backward pass on the last layer
    wname, bname = self._get_wb_names(self.num_conv + self.num_affine + 1)
    dH, dw, db = affine_backward(dout, caches[self.num_conv + self.num_affine])
    grads[wname] = dw
    grads[bname] = db
    # Add regularization loss for this weight matrix
    loss += 0.5 * self.reg * np.sum(np.square(self.params[wname]))
    grads[wname] += self.reg * self.params[wname]
    
    # Second, perform backward pass on intermediate affine layers
    for i in range(self.num_conv + self.num_affine, self.num_conv, -1):
        wname, bname = self._get_wb_names(i)
        # Calculate gradient
        cache = caches[i -1]
        if self.use_batchnorm:
            gammaname, betaname = self._get_bn_names(i)
            dH, dw, db, dgamma, dbeta = affine_bnorm_relu_backward(dH, (cache))
            grads[gammaname] = dgamma
            grads[betaname] = dbeta
        else:
            dH, dw, db = affine_relu_backward(dH, (cache))
        grads[wname] = dw
        grads[bname] = db
        # Add regularization loss for this weight matrix
        loss += 0.5 * self.reg * np.sum(np.square(self.params[wname]))
        grads[wname] += self.reg * self.params[wname]

    # Finally, perform backward pass on conv layers
    for i in range(self.num_conv, 0, -1):
        wname, bname = self._get_wb_names(i)
        # Calculate gradient
        cache = caches[i -1]
        if self.use_batchnorm:
            gammaname, betaname = self._get_bn_names(i)
            dH, dw, db, dgamma, dbeta = \
                    conv_bnorm_relu_pool_backward(dH, (cache))
            grads[gammaname] = dgamma
            grads[betaname] = dbeta
        else:
            dH, dw, db = conv_relu_pool_backward(dH, (cache))
        grads[wname] = dw
        grads[bname] = db
        # Add regularization loss for this weight matrix
        loss += 0.5 * self.reg * np.sum(np.square(self.params[wname]))
        grads[wname] += self.reg * self.params[wname]

    return loss, grads
