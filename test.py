from scalar import Scalar
import scalar
from matrix import Matrix
import autodiff
import math

num_tests = 0
num_passed = 0

def test_assert(cond, message=None):
  global num_tests, num_passed

  num_tests += 1
  assert(cond)
    
  if cond:
    num_passed += 1
  elif message is not None:
      print("Failure:", message)

def test_simple():
  test_assert(3.0 == (Scalar(1) + Scalar(2)).value, "1 + 2 == 3")
  test_assert(3.0 == (1 + Scalar(2)).value, "1 + 2 == 3")
  test_assert(3.0 == (Scalar(1) + 2).value, "1 + 2 == 3")

  test_assert(3.5 == (Scalar(.5) * Scalar(7)).value, ".5 * 7 = 3.5")
  test_assert(3.5 == (.5 * Scalar(7)).value, ".5 * 7 = 3.5")
  test_assert(3.5 == (Scalar(.5) * 7).value, ".5 * 7 = 3.5")

  test_assert(-1 == (Scalar(1) - Scalar(2)).value, "1 - 2 == -1")
  test_assert(-1 == (1 - Scalar(2)).value, "1 - 2 == -1")
  test_assert(-1 == (Scalar(1) - 2).value, "1 - 2 == -1")

  test_assert(3.4 == (Scalar(17) / Scalar(5)).value, "17 / 5 == 3.4")
  test_assert(3.4 == (17 / Scalar(5)).value, "17 / 5 == 3.4")
  test_assert(3.4 == (Scalar(17) / 5).value, "17 / 5 == 3.4")

  test_assert(81 == (Scalar(3) ** Scalar(4)).value, "3^4 == 81")
  test_assert(81 == (3 ** Scalar(4)).value, "3^4 == 81")
  test_assert(81 == (Scalar(3) ** 4).value, "3^4 == 81")

  test_assert(-17 == (-Scalar(17)).value, "-17")

def func_constant(x):
  return Scalar(3) if isinstance(x, Scalar) else 3

def func_linear(x):
  return 3*x + 2

def func_add(x):
  return 3*x + x

def func_sub(x):
  return 3*x - x

def func_pow(x):
  return 3*x**2 + 2

def func_mul(x):
  return x*x

def func_div(x):
  return 3/(x + 10)

def func_neg(x):
  return -x

def func_add_chain(x):
  return 3*(x*x + 1/(x+10)) + (x*x + 1/(x+10))

def func_sub_chain(x):
  return 3*(x*x + 1/(x+10)) - (x*x + 1/(x+10))

def func_pow_chain(x):
  return 3*(x*x + 1/(x+10))**2 + 2

def func_mul_chain(x):
  return (x*x + 1/(x+10))*(x*x + 1/(x+10))

def func_div_chain(x):
  return 3/((x*x + 1/(x+10)) + 10)

def func_neg_chain(x):
  return -(x*x + 1/(x+10))

def func_exp(x):
  return math.exp(1)**(-x)

def func_sigmoid(x):
  return 1.0 / (1.0 + math.exp(1)**(-x))

def func_log(x):
  return scalar.scalar_log(x)

def test_derivative(func, low_range, high_range):
  for i in range(10):
    x = low_range + i * (high_range - low_range) / 10
    eval_native = func(x)
    
    # Finite difference
    _, finite_diff_derivative = autodiff.finite_difference(func, [x])
    
    # Forward mode autodiff
    eval_fwd, forward_deriv, forward_ops = autodiff.compute_gradients(func, [x], 0, reverse_mode = False)
    test_assert(eval_native == eval_fwd)
    test_assert(abs(finite_diff_derivative - forward_deriv) < 1e-3)
    
    # Reverse mode autodiff
    eval_reverse, reverse_deriv, reverse_ops = autodiff.compute_gradients(func, [x], 0, reverse_mode = True)
    test_assert(eval_native == eval_reverse)
    test_assert(abs(finite_diff_derivative - reverse_deriv) < 1e-3)

    test_assert(reverse_ops <= forward_ops)

def test_simple_derivative():
  test_derivative(func_constant, -2.0, 2.0)
  test_derivative(func_add, -2.0, 2.0)
  test_derivative(func_sub, -2.0, 2.0)
  test_derivative(func_pow, -2.0, 2.0)
  test_derivative(func_mul, -2.0, 2.0)
  test_derivative(func_div, -2.0, 2.0)
  test_derivative(func_neg, -2.0, 2.0)
  test_derivative(func_linear, -2.0, 2.0)
  test_derivative(func_add_chain, -2.0, 2.0)
  test_derivative(func_sub_chain, -2.0, 2.0)
  test_derivative(func_pow_chain, -2.0, 2.0)
  test_derivative(func_mul_chain, -2.0, 2.0)
  test_derivative(func_div_chain, -2.0, 2.0)
  test_derivative(func_neg_chain, -2.0, 2.0)
  test_derivative(func_exp, -2.0, 2.0)
  test_derivative(func_sigmoid, -2.0, 2.0)
  test_derivative(func_log, .01, 2.0)

def func_gradient_const(x):
  p = Matrix(3, 1, [[0], [0], [0]])
  b = Matrix(1, 1, [[12]])
  return x.transpose().matmul(p) + (b)

def func_gradient_dot(x):
  p = Matrix(3, 1, [[3.0], [4.0], [5.0]])
  return x.transpose().matmul(p)

def func_gradient_matmul(x):
  p = Matrix(3, 1, [[3.0], [4.0], [5.0]])
  z = x.matmul(p.transpose()).reduce_sum()
  return Matrix(1,1,[[z]])

def func_gradient_hadamard(x):
  p = Matrix(3, 1, [[3.0], [4.0], [5.0]])
  z = (x * p).reduce_sum()
  return Matrix(1,1,[[z]])

def func_gradient_square(x):
  z = (x * x).reduce_sum()
  return Matrix(1,1,[[z]])

def func_gradient_scalarmul(x):
  z = (12 * x).reduce_sum()
  return Matrix(1,1,[[z]])

def test_gradient(func, low_range, high_range):
  data = [[0],[0],[0]]
  for i in range(3):
    data[0][0] = low_range + i * (high_range - low_range) / 3
    for j in range(3):
      data[1][0] = low_range + j * (high_range - low_range) / 3
      for k in range(3):
        data[2][0] = low_range + k * (high_range - low_range) / 3
        
        x = Matrix(3, 1, data)
        
        eval_native = func(x)
  
        _, finite_diff_derivative = autodiff.finite_difference(func, [x])
  
        # Forward mode autodiff
        eval_fwd, forward_grad, forward_ops = autodiff.compute_gradients(func, [x], 0, reverse_mode = False)
        test_assert(eval_native.compare(eval_fwd))
        test_assert(finite_diff_derivative.compare(forward_grad, 1e-3))
  
        # Reverse mode autodiff  
        eval_reverse, reverse_grad, reverse_ops = autodiff.compute_gradients(func, [x], 0, reverse_mode = True)
        test_assert(eval_native.compare(eval_reverse))
        test_assert(finite_diff_derivative.compare(reverse_grad, 1e-3))
  
        test_assert(reverse_ops <= forward_ops)
  
def test_gradients():
  test_gradient(func_gradient_dot, -1.0, 2.0)
  test_gradient(func_gradient_const, -1.0, 2.0)
  test_gradient(func_gradient_matmul, -1.0, 2.0)
  test_gradient(func_gradient_hadamard, -1.0, 2.0)
  test_gradient(func_gradient_square, -1.0, 2.0)
  test_gradient(func_gradient_scalarmul, -1.0, 2.0)

def func_jacobian_matmul(x):
  p = Matrix(3, 3, [[3.0, 4.5, 11.5], [4.0, 1.5, 9], [5.0, 1, 2]])
  return p.matmul(x)

def func_jacobian_const(x):
  return Matrix(3, 1, [[3.0], [4.5], [11.5]]) + x - x
  
def test_jacobians():
  x = Matrix(3, 1, [[4.0], [7.0], [2.0]])
  correct = Matrix(3, 3, [[3.0, 4.5, 11.5], [4.0, 1.5, 9], [5.0, 1, 2]]).transpose()

  _, jacobian = autodiff.finite_difference(func_jacobian_matmul, [x])
  test_assert(correct.compare(jacobian, 1e-3))
  _, jacobian, _ = autodiff.compute_gradients(func_jacobian_matmul, [x], 0, reverse_mode = True)
  test_assert(correct.compare(jacobian, 1e-3))
  _, jacobian, _ = autodiff.compute_gradients(func_jacobian_matmul, [x], 0, reverse_mode = False)
  test_assert(correct.compare(jacobian, 1e-3))

  correct = Matrix(1, 3, [[0, 0, 0]])
  x = 2.5
  _, jacobian = autodiff.finite_difference(func_jacobian_const, [x])
  test_assert(correct.compare(jacobian, 1e-3))
  _, jacobian, _ = autodiff.compute_gradients(func_jacobian_const, [x], 0, reverse_mode = True)
  test_assert(correct.compare(jacobian, 1e-3))
  _, jacobian, _ = autodiff.compute_gradients(func_jacobian_const, [x], 0, reverse_mode = False)
  test_assert(correct.compare(jacobian, 1e-3))
  
def main():
  global num_tests, num_passed

  test_simple()
  test_simple_derivative()
  test_gradients()
  test_jacobians()

  if num_tests > num_passed:
    print("%d FAILED!" % (num_tests - num_passed))
  print("%d / %d passed" % (num_passed, num_tests))
  
if __name__ == "__main__":
  main()

