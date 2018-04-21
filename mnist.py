import numpy as np
import random
from matrix import Matrix
import autodiff
import math
import scalar
from tensorflow.examples.tutorials.mnist import input_data
import skimage.transform

def load_mnist_dataset(train_xs, train_ys):
  #train_xs = train_xs[:5000,:]
  #train_ys = train_ys[:5000,:]
  
  # resize
  resized = np.zeros((train_xs.shape[0], 7, 7))
  for i in range(train_xs.shape[0]):
    fullim = train_xs[i].copy()
    fullim.resize((28,28))
    resized[i] = skimage.transform.downscale_local_mean(fullim, (4, 4))

  resized.resize(resized.shape[0], 7*7)
  train_xs = resized
  
  train_xs = Matrix(train_xs.shape[0], train_xs.shape[1], train_xs.tolist())
  train_ys = Matrix(train_ys.shape[0], train_ys.shape[1], train_ys.tolist())
  return train_xs, train_ys

def load_mnist():
  mn = input_data.read_data_sets("MNIST_data/", one_hot=True)
  train_xs, train_ys = load_mnist_dataset(mn.train._images, mn.train._labels)
  test_xs, test_ys = load_mnist_dataset(mn.test._images, mn.test._labels)
  return train_xs, train_ys, test_xs, test_ys

def softmax(x):
  m = x.apply_copy(lambda s: math.exp(1)**s)
  s = m.reduce_sum()
  return m / s

def linear_model(params, x):
  oldr = params.rows
  oldc = params.cols
  params.rows = 10
  params.cols = oldr // 10
  W = params
  y = W.matmul(x)
  y = softmax(y)
  params.rows = oldr
  params.cols = oldc
  return y

def error_linear_model(params, x, y_target):
  y = linear_model(params, x)
  loss = 0
  for r in range(y.rows):
    loss += -y_target[r] * scalar.scalar_log(y[r] + 1e-8)
  return loss

def error_batch_linear_model(params, xs, ys):
  total_error = 0
  for i in range(xs.rows):
    total_error += error_linear_model(params, xs[i].transpose(), ys[i].transpose())
  return total_error / len(range(xs.rows))

def eval_model(params, xs, ys):
  correct = 0
  for i in range(xs.rows):
    prediction = linear_model(params, xs[i].transpose())
    max_value = -1
    predicted_label = 0
    for j in range(prediction.rows):
      if prediction[j] > max_value:
        max_value = prediction[j]
        predicted_label = j
    if ys[i,predicted_label] == 1.0:
      correct += 1
  return correct / xs.rows

def train_linear_model(xs, ys, test_xs, test_ys):
  params = Matrix(xs.cols * ys.cols, 1)

  batch_size = 100

  print('Initial accuracy', eval_model(params, test_xs, test_ys))

  batch_indices = [i for i in range(xs.rows)]
  
  momentum = Matrix(xs.cols, 1)
  
  step_size = .5
  for epoch in range(5):
    random.shuffle(batch_indices)
    num_batches = len(batch_indices) // batch_size
    for i in range(num_batches):
      batch = batch_indices[i*batch_size:(i*batch_size + batch_size)]
      
      batch_xs = xs.gather_rows(batch)
      batch_ys = ys.gather_rows(batch)
      
      f_val, f_grad, opcount = autodiff.compute_gradients(error_batch_linear_model, [params, batch_xs, batch_ys], 0, reverse_mode = True)
      #f_val, f_grad = autodiff.f_d(error_batch_linear_model, [params, batch_xs, batch_ys], 0)
      
      #momentum = 0.9*momentum + f_grad * -step_size
      params += f_grad * -step_size
      if (i + 1) % 10 == 0:
        print("Epoch %d, Batch %d (of %d) Error %f  e_grad_norm=%f (opcount=%d)" % (epoch + 1, i + 1, num_batches, f_val, f_grad.euclidean_norm(), opcount))

    accuracy = eval_model(params, test_xs, test_ys)
    print("Epoch %d accuracy %f" % (epoch + 1, accuracy))

def main():  
  print("Loading data...")
  train_xs, train_ys, test_xs, test_ys = load_mnist()
  
  print("Training")
  train_linear_model(train_xs, train_ys, test_xs, test_ys)


if __name__ == "__main__":
  main()
