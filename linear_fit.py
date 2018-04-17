from matrix import Matrix
import random
import autodiff

def f_line(params, x):
  return params[0] * x + params[1]

def error_line(params, x, y_target):
  y = f_line(params, x)
  return (y - y_target)**2

def error_batch(params, xs, ys):
  total_error = 0
  for i in range(xs.rows):
    e = error_line(params, xs[i], ys[i])
    total_error += e
  return total_error / len(range(xs.rows))

def gen_data(m, b, num_points=100):
  xs = Matrix(num_points, 1)
  ys = Matrix(num_points, 1)
  for i in range(num_points):
    xs[i] = random.uniform(-5, 5)
    ys[i] = m * xs[i] + b + random.gauss(0, .2)
  return xs, ys

def main():
  target_m = .7
  target_b = 4
  xs, ys = gen_data(target_m, target_b)
  params = Matrix(2, 1, [[-2], [0]])

  for i in range(1000):
    f_val, f_grad, opcount = autodiff.compute_gradients(error_batch, [params, xs, ys], 0, reverse_mode = True)
    params += f_grad * -4e-3
    if (i+1) % 10 == 0:
      print("%d: Error %f  e_grad=%f  m=%f  b=%f (opcount=%d)" % (i + 1, f_val, f_grad.reduce_sum(), params[0], params[1], opcount))

  print("Final m=%f, b=%f" % (params[0], params[1]))  
  print("Compared to target m=%f, b=%f" % (target_m, target_b))  
if __name__ == "__main__":
  main()

