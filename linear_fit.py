from matrix import Matrix
import random
from autodiff import convert_to_number
from number import Number

def f_line(params, x):
  m = Matrix(1, 2, [[x, 1.0]])
  return m.matmul(params)

def error_line(params, x, y_target):
  y = f_line(params, x)
  e = (y.get(0,0) - y_target)**2
  m = Matrix(1, 1, [[e]])
  return m

def error_batch(params, training_pairs):
  total_error = Matrix(1,1,0)
  for x, y_target in training_pairs:
    e = error_line(params, x, y_target)
    total_error = total_error.add(e)
  return total_error.scalarmul(1.0/len(training_pairs))

def gen_data(m, b, num_points=100):
  training_pairs = []
  for i in range(num_points):
    x = random.uniform(-5, 5)
    y = m*x + b + random.gauss(0, .2)
    training_pairs.append((Number(x), Number(y)))
  return training_pairs



def main():
  training_pairs = gen_data(.7, 4)
  params = Matrix(2, 1, [[-2], [0]])
  params = convert_to_number(params)
  
  #import pdb;pdb.set_trace()
  for i in range(1000):
    f_val = error_batch(params, training_pairs)
    f_grad = f_val.reverse_autodiff(params)
    params = params.add(f_grad.scalarmul(-4e-3))
    params = Matrix(2,1,[[params.get(0,0).value],[params.get(0,1).value]])
    params = convert_to_number(params)
    if (i+1) % 10 == 0:
      print("%d: Error %s  e_grad = %s  m = %s  b = %s" % (i + 1, f_val.get(0,0), f_grad.reduce_sum(), params.get(0,0), params.get(0,1)))
  
if __name__ == "__main__":
  main()

