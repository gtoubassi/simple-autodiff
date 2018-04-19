from matrix import Matrix
import matrix
from scalar import Scalar

epsilon = 1e-7

def _scalarize(m):
  if isinstance(m, Matrix):
    assert m.rows == 1
    assert m.cols == 1
    return m.get(0,0)
  return m

def finite_difference(func, params, gradient_target = 0):
  global epsilon

  derivatives = []

  # compute f(x)
  eval_f = _scalarize(func(*params))
  
  #compute f(x+h)
  i = gradient_target
  param = params[i]
  if isinstance(param, Matrix):
    assert param.cols == 1, "can only take finite diff gradients of scalar functions (no jacobians)"
    deriv = [[0]*param.cols for x in range(param.rows)]
    for r in range(param.rows):
      for c in range(param.cols):
        epsilon_param = Matrix(param.rows, param.cols, lambda row, col: param.get(row, col) + epsilon if row == r and col == c else param.get(row, col))
        epsilon_params = list(params)
        epsilon_params[i] = epsilon_param
        eval_f_epsilon = _scalarize(func(*epsilon_params))
        deriv[r][c] = (eval_f_epsilon - eval_f) / epsilon
    deriv = Matrix(param.rows, param.cols, deriv)
    derivatives.append(deriv)
  else:
    epsilon_params = list(params)
    epsilon_params[i] += epsilon
    eval_f_epsilon = func(*epsilon_params)
    deriv = (eval_f_epsilon - eval_f) / epsilon
    derivatives.append(deriv)

  if len(derivatives) == 1:
    derivatives = derivatives[0]

  return (eval_f, derivatives) 


def reverse_autodiff(result, var):
  Scalar.opcount = 0

  if isinstance(result, Matrix):
    result.apply(lambda x: x._reset_grad())
    for i in range(len(result.data)):
      result.data[i].grad_value = 1
  else:
    result._reset_grad()
    result.grad_value = 1
  
  if isinstance(var, Matrix):
    var.apply(lambda x: x._reset_grad())
    var.apply(lambda x: x._reverse_autodiff())
    z = Matrix(var.rows, 1)
    for r in range(z.rows):
      z.data[r * z.cols + 0] = var.get(r, 0).grad_value
  else:
    var._reset_grad()
    z = var._reverse_autodiff()    
  # Todo, we are assuming a true gradient, no jacobians here
  return z

  def forward_autodiff(self, var):
    Scalar.opcount = 0
    return z
  

def forward_autodiff(result, var):
  Scalar.opcount = 0

  # Todo, we are assuming a true gradient, no jacobians here
  if isinstance(result, Matrix):
    assert result.rows == 1 and result.cols == 1, 'no jacobian support'
    result = result[0,0]
  
  if isinstance(var, Matrix):
    z = Matrix(var.rows, 1)
    for r in range(z.rows):
      result._reset_grad()
      var.apply(lambda x: x._reset_grad())
      var.data[r * z.cols + 0].grad_value = 1
      z.data[r * z.cols + 0] = result._forward_autodiff()
  else:
    result._reset_grad()
    var._reset_grad()
    var.grad_value = 1
    z = result._forward_autodiff()
    
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
