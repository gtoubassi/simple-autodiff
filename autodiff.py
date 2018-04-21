from matrix import Matrix
import matrix
from scalar import Scalar


def _scalarize(m):
  if isinstance(m, Matrix):
    assert m.rows == 1
    assert m.cols == 1
    return m.get(0,0)
  return m

def _matricize(s):
  if isinstance(s, Matrix):
    return s
  return Matrix(1, 1, s)

def is_scalarish(val):
  return isinstance(val, float) or isinstance(val, int) or isinstance(val, Scalar)

def finite_difference(func, params, gradient_target = 0):
  epsilon = 1e-7

  # compute f(x)
  result = func(*params)

  var = params[gradient_target]

  result_was_scalar = is_scalarish(result)
  var_was_scalar = is_scalarish(var)
  var = _matricize(var)
  result = _matricize(result)

  assert isinstance(var, Matrix) and var.cols == 1
  assert isinstance(result, Matrix) and result.cols == 1
  
  z = Matrix(var.rows, result.rows)
  for r in range(z.rows):
    for c in range(z.cols):
      epsilon_var = var.copy()
      epsilon_var[r] += epsilon
      epsilon_params = list(params)
      epsilon_params[gradient_target] = _scalarize(epsilon_var) if var_was_scalar else epsilon_var
      result_epsilon = _matricize(func(*epsilon_params))
      z[r] = ((result_epsilon - result) / epsilon).transpose()

  if result_was_scalar and z.rows == z.cols == 1:
    z = z[0, 0]
    result = result[0, 0]

  return (result, z) 

def reverse_autodiff(result, var):
  Scalar.opcount = 0

  result_was_scalar = is_scalarish(result)
  var = _matricize(var)
  result = _matricize(result)

  assert isinstance(var, Matrix) and var.cols == 1
  assert isinstance(result, Matrix) and result.cols == 1

  z = Matrix(var.rows, result.rows)
  for r in range(z.cols):
    result._apply(lambda x: x._reset_grad())
    result[r].grad_value = 1

    var._apply(lambda x: x._reset_grad())
    var._apply(lambda x: x._reverse_autodiff())
    for c in range(var.rows):
      z[c,r] = var[c].grad_value
  
  if result_was_scalar and z.rows == z.cols == 1:
    z = z[0,0]

  return z

def forward_autodiff(result, var):
  Scalar.opcount = 0

  result_was_scalar = is_scalarish(result)
  var = _matricize(var)
  result = _matricize(result)

  assert isinstance(var, Matrix) and var.cols == 1
  assert isinstance(result, Matrix) and result.cols == 1
  
  z = Matrix(var.rows, result.rows)
  for r in range(z.rows):
    result._apply(lambda x: x._reset_grad())
    var._apply(lambda x: x._reset_grad())
    var[r].grad_value = 1
    for c in range(result.rows):
      z[r, c] = result[c, 0]._forward_autodiff()
  
  if result_was_scalar and z.rows == z.cols == 1:
    z = z[0,0]

  return z

def compute_gradients(func, args, gradient_target, reverse_mode = True):
  args = list(map((lambda x: matrix.convert_to_scalar(x)), args))
  f_val = func(*args)
  if reverse_mode:
    f_grad = reverse_autodiff(f_val, args[gradient_target])
  else:
    f_grad = forward_autodiff(f_val, args[gradient_target])
  f_val = matrix.convert_from_scalar(f_val)
  f_grad = matrix.convert_from_scalar(f_grad)
  return f_val, f_grad, Scalar.opcount
