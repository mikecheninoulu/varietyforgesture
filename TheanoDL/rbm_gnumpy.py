# this is a modified version of the example script that comes with cudamat

def test_gnumpy(dat, num_epochs):
  import gnumpy as gpu
  import numpy 
  import time
  # load data. <dat> is 2 dimensional: 60000 X 784
  #dat = gpu.garray(load('mnist_cudaTest').T/255.) 
  # training parameters
  epsilon = 0.1
  momentum = 0.9
  batch_size = 128
  num_batches = dat.shape[0]/batch_size
  # model parameters
  num_vis = dat.shape[1]
  num_hid = 4096
  # initialize weights
  w_vh = 0.1 * gpu.randn(num_vis, num_hid)
  w_v = gpu.zeros(num_vis)
  w_h = -4. * gpu.ones(num_hid)
  # initialize weight updates
  wu_vh = gpu.zeros((num_vis, num_hid))
  wu_v = gpu.zeros(num_vis)
  wu_h = gpu.zeros(num_hid)
  for epoch in range(num_epochs):
    err = []
    tic = time.clock()
    for batch in range(num_batches):
      # positive phase
      v1 = dat[batch*batch_size : (batch + 1)*batch_size]
      h1 = (gpu.dot(v1, w_vh) + w_h).logistic()
      # sample hiddens
      hSampled = h1.rand() < h1
      # negative phase
      v2 = (gpu.dot(hSampled, w_vh.T) + w_v).logistic()
      h2 = (gpu.dot(v2, w_vh) + w_h).logistic()
      # update weights
      wu_vh = wu_vh * momentum + gpu.dot(v1.T, h1) - gpu.dot(v2.T, h2)
      wu_v = wu_v * momentum + v1.sum(0) - v2.sum(0)
      wu_h = wu_h * momentum + h1.sum(0) - h2.sum(0)
      
      w_vh += wu_vh * (epsilon/batch_size)
      w_v += wu_v * (epsilon/batch_size)
      w_h += wu_h * (epsilon/batch_size)
      # calculate reconstruction error
      err.append((v2-v1).euclid_norm()**2/(num_vis*batch_size))
    toc = time.clock()
    print "Mean squared error: %.4f, takes time: %d" % (numpy.mean(err), toc-tic)
  return w_vh, w_v, w_h


def test_cpu_numpy(dat, num_epochs):
  import numpy 
  import time
  logistic = lambda x:1.0 / (1.0 + numpy.exp(-1.0 * x))   
  epsilon = 0.1
  momentum = 0.9
  batch_size = 128
  num_batches = dat.shape[0]/batch_size
  # model parameters
  num_vis = dat.shape[1]
  num_hid = 4096
  # initialize weights
  w_vh = 0.1 * numpy.random.randn(num_vis, num_hid)
  w_v = numpy.zeros(num_vis)
  w_h = -4. * numpy.ones(num_hid)
  # initialize weight updates
  wu_vh = numpy.zeros((num_vis, num_hid))
  wu_v = numpy.zeros(num_vis)
  wu_h = numpy.zeros(num_hid)
  for epoch in range(num_epochs):
    err = []
    tic = time.clock()
    for batch in range(num_batches):
      # positive phase
      v1 = dat[batch*batch_size : (batch + 1)*batch_size]
      h1 = logistic(numpy.dot(v1, w_vh) + w_h)
      # sample hiddens
      hSampled = numpy.random.rand(h1.shape[0], h1.shape[1]) < h1
      # negative phase
      v2 = logistic(numpy.dot(hSampled, w_vh.T) + w_v)   
      h2 = logistic(numpy.dot(v2, w_vh) + w_h)
      # update weights
      wu_vh = wu_vh * momentum + numpy.dot(v1.T, h1) - numpy.dot(v2.T, h2)
      wu_v = wu_v * momentum + v1.sum(0) - v2.sum(0)
      wu_h = wu_h * momentum + h1.sum(0) - h2.sum(0)
      
      w_vh += wu_vh * (epsilon/batch_size)
      w_v += wu_v * (epsilon/batch_size)
      w_h += wu_h * (epsilon/batch_size)
      # calculate reconstruction error
      err.append(sum(sum(v2-v1)**2)/(num_vis*batch_size))
    toc = time.clock()
    print "Mean squared error: %.4f, takes time: %d" % (numpy.mean(err), toc-tic)
  return w_vh, w_v, w_h


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp**(-1.0 * z))
    return s
