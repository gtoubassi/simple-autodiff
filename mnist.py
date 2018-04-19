import numpy as np
import imageio
from matrix import Matrix
import random
import autodiff
import math
import scalar

imdim = 7

def image_to_matrix(image, m, imrow):
  for row in range(image.shape[0]):
    for col in range(image.shape[1]):
      m[imrow, row*image.shape[0] + col] = image[col, row]
  return m

def get_dataset(image, labels):
  w = image.shape[0]
  h = image.shape[1]

  # count how many 0s and 1s?
  num_final = sum(map(lambda x: 1 if x < 50 else 0, labels))
  num_final = min(10*32, num_final)

  ys = Matrix(num_final, 1)
  xs = Matrix(num_final, imdim*imdim + 1, 1.0)  # +1 column and the 1.0 default for the bias term

  i = 0
  n = 0
  for y in range(0, h, imdim):
    for x in range(0, w, imdim):
      if labels[i] < 50:
        digit_image = (image[y:(y+imdim), x:(x+imdim)].astype(float) / 255.0) ** 2
        image_to_matrix(digit_image, xs, n)
        ys[n] = labels[i] - 48
        n += 1
        if n == num_final:
          return xs, ys
      i += 1

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
  # −(yln(p)+(1−y)ln(1−p))
  #return (y - y_target)**2

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

def train_linear_model(xs, ys):
  params = Matrix(xs.cols, 1, lambda r, c: random.gauss(0,.1))

  batch_size = 32

  print('Accuracy', eval_model(params, xs, ys))
  
  batch_indices = [i for i in range(xs.rows)]
  
  momentum = Matrix(xs.cols, 1)
  
  step_size = .01
  for epoch in range(1000):
    random.shuffle(batch_indices)
    for i in range(0, len(batch_indices), batch_size):
      batch = batch_indices[i:(i + batch_size)]
      
      batch_xs = xs.gather_rows(batch)
      batch_ys = ys.gather_rows(batch)
      
      f_val, f_grad, opcount = autodiff.compute_gradients(error_batch_linear_model, [params, batch_xs, batch_ys], 0, reverse_mode = True)
      #f_val, f_grad = autodiff.f_d(error_batch_linear_model, [params, batch_xs, batch_ys], 0)
      
      momentum = 0.9*momentum + f_grad * -step_size
      params += momentum
      #print("Epoch %d, Batch %d Error %f  e_grad_norm=%f (opcount=%d)" % (epoch, i + 1, f_val, f_grad.euclidean_norm(), opcount))

    #if (epoch + 1) % 25 == 0:
    #  step_size /= 2
    #  print('step:', step_size)

    f_val, f_grad, _ = autodiff.compute_gradients(error_batch_linear_model, [params, xs, ys], 0, reverse_mode = True)
    accuracy = eval_model(params, xs, ys)
    print("Epoch %d accuracy %f Error %f e_grad_norm=%f" % (epoch, accuracy, f_val, f_grad.euclidean_norm()))

def main():
  print("Loading data...")
  train_image = imageio.imread('data/train_digits%dx%d.png' % (imdim, imdim))
  test_image = imageio.imread('data/test_digits%dx%d.png' % (imdim, imdim))

  with open("data/train_digits.txt", "rb") as f:
    train_labels = f.read()
  with open("data/test_digits.txt", "rb") as f:
    test_labels = f.read()
  
  train_xs, train_ys = get_dataset(train_image, train_labels)
  test_xs, test_ys = get_dataset(test_image, test_labels)
  
  print("Training")
  train_linear_model(train_xs, train_ys)
  #print(train_xs.rows, test_ys.rows)
  #dump_data(test_xs, test_ys)


if __name__ == "__main__":
  main()
