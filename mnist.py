import numpy as np
import random
from matrix import Matrix
import autodiff
import math
import scalar
from tensorflow.examples.tutorials.mnist import input_data
import skimage.transform

def load_mnist_dataset(orig_images, orig_labels):
  idxs = []
  new_labels = []
  for i in range(orig_labels.shape[0]):
    if orig_labels[i,0] == 1:
      new_labels.append([0.0])
      idxs.append(i)
    elif orig_labels[i,1] == 1:
      new_labels.append([1.0])
      idxs.append(i)

  train_xs = orig_images.take(idxs, axis=0)
  train_ys = np.asarray(new_labels)

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

def dump_data(xs, ys, limit = 250):
  for i in range(min(xs.rows, limit)):
    print('Label', ys[i])
    for row in range(imdim):
      for col in range(imdim):
        print('*' if xs[i, col*imdim + row] < .7 else ' ', end='')
      print()
    print()

def sigmoid(x):
  e = math.exp(1)
  return 1 / (1 + e**(-x))

def linear_model(params, x):
  dot = x.matmul(params)[0]
  return sigmoid(dot)

def error_linear_model(params, x, y_target):
  y = linear_model(params, x)
  return -(y_target * scalar.scalar_log(y + 1e-8) + (1-y_target) * scalar.scalar_log(1 - y + 1e-8))

def error_batch_linear_model(params, xs, ys):
  total_error = 0
  for i in range(xs.rows):
    e = error_linear_model(params, xs[i], ys[i])
    total_error += e
  return total_error / len(range(xs.rows))

def eval_model(params, xs, ys):
  correct = 0
  for i in range(xs.rows):
    prediction = linear_model(params, xs[i])
    predicted_label = 1 if prediction > .5 else 0
    if predicted_label == ys[i]:
      correct += 1
  return correct / xs.rows

def train_linear_model(xs, ys, test_xs, test_ys):
  params = Matrix(xs.cols, 1)

  batch_size = 32

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
