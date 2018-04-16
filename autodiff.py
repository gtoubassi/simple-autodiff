from matrix import Matrix
from number import Number

epsilon = 1e-7

def _scalarize(m):
  if isinstance(m, Matrix):
    assert m.rows == 1
    assert m.cols == 1
    return m.get(0,0)
  return m

def finite_difference(func, params):
  global epsilon

  derivatives = []

  # compute f(x)
  eval_f = _scalarize(func(*params))
  
  #compute f(x+h)
  for i, param in enumerate(params):
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

def convert_to_number(m):
  if isinstance(m, Number):
    return m
  if isinstance(m, Matrix):
    return Matrix(m.rows, m.cols, lambda r, c: Number(m.get(r, c)))
  return Number(m)
  
def autodiff(func, params, reverse_mode = True):
  global epsilon

  derivatives = []

  p = []
  for param in params:
    p.append(convert_to_number(param))
  params = p

  # compute f(x)
  eval_f = func(*params)

  opcount = 0
  for param in params:
    if reverse_mode:
      deriv = eval_f.reverse_autodiff(param)
    else:
      deriv = eval_f.forward_autodiff(param)
    opcount += Number.opcount
    derivatives.append(deriv)

  if len(derivatives) == 1:
    derivatives = derivatives[0]

  return (eval_f, derivatives, opcount) 
