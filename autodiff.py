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


def reverse_autodiff(result, var):
  Number.opcount = 0

  if isinstance(result, Matrix):
    result.apply(lambda x: x._reset_grad())
    for i in range(len(result.data)):
      result.data[i].grad_value = 1
  else:
    result._reset_grad()
    result.grad_value = 1
  
  if isinstance(var, Matrix):
    var.apply(lambda x: x._reset_grad())
    var.apply(lambda x: x._do_reverse_autodiff())
    z = Matrix(var.rows, 1)
    for r in range(z.rows):
      z.data[r * z.cols + 0] = var.get(r, 0).grad_value
  else:
    var._reset_grad()
    z = var._do_reverse_autodiff()    
  # Todo, we are assuming a true gradient, no jacobians here
  return z

  def forward_autodiff(self, var):
    Number.opcount = 0
    return z
  

def forward_autodiff(result, var):
  Number.opcount = 0

  # Todo, we are assuming a true gradient, no jacobians here
  if isinstance(result, Matrix):
    assert results.rows == 1 and result.cols == 1, 'no jacobian support'
    result = result[0,0]
  
  if isinstance(var, Matrix):
    z = Matrix(var.rows, 1)
    for r in range(z.rows):
      result._reset_grad()
      var.apply(lambda x: x._reset_grad())
      var.data[r * z.cols + 0].grad_value = 1
      z.data[r * z.cols + 0] = result._do_forward_autodiff()
  else:
    result._reset_grad()
    var._reset_grad()
    var.grad_value = 1
    z = result._do_forward_autodiff()
    
  return z

def convert_to_number(m):
  if isinstance(m, Number):
    return m
  if isinstance(m, Matrix):
    return Matrix(m.rows, m.cols, lambda r, c: Number(m.get(r, c)))
  return Number(m)
  
def convert_from_number(m):
  if isinstance(m, Number):
    return m.value
  if isinstance(m, Matrix):
    return Matrix(m.rows, m.cols, lambda r, c: m[r,c].value if isinstance(m[r,c], Number) else m[r,c])
  return m

def compute_gradients(func, args, gradient_target, reverse_mode = True):
  args = list(map((lambda x: convert_to_number(x)), args))
  f_val = func(*args)
  if reverse_mode:
    f_grad = reverse_autodiff(f_val, args[gradient_target])
  else:
    f_grad = forward_autodiff(f_val, args[gradient_target])
  f_val = convert_from_number(f_val)
  f_grad = convert_from_number(f_grad)
  return f_val, f_grad, Number.opcount
