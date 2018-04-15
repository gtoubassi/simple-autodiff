from num import Number

class Matrix:
  
  def __init__(self, rows, cols, initializer = 0.0):
    self.rows = rows
    self.cols = cols
    self.data = [0.0] * (rows * cols)
    if callable(initializer):
      for r in range(self.rows):
        for c in range(self.cols):
          self.data[r * self.cols + c] = initializer(r, c)
    elif isinstance(initializer, list):
      for r in range(self.rows):
        for c in range(self.cols):
          self.data[r * self.cols + c] = initializer[r][c]
    elif isinstance(initializer, float) or isinstance(initializer, int):
      self.data = [initializer] * (rows * cols)

  def reverse_autodiff(self, wrt):
    Number.opcount = 0
    self.apply(lambda x: x._reset_grad())
    wrt.apply(lambda x: x._reset_grad())
    for i in range(len(self.data)):
      self.data[i].grad_value = 1
    wrt.apply(lambda x: x._do_reverse_autodiff())
    z = Matrix(wrt.rows, 1)
    # Todo, we are assuming a true gradient, no jacobians here
    for r in range(z.rows):
      z.data[r * z.cols + 0] = wrt.get(r, 0)._do_reverse_autodiff()
    return z

  def apply(self, func):
    for i in range(len(self.data)):
      func(self.data[i])
    
  def compare(self, other, tolerance = 0.0):
    if self.rows != other.rows or self.cols != other.cols:
      return False
    for i in range(len(self.data)):
      if abs(self.data[i] - other.data[i]) > tolerance:
        return False
    return True
      
  def get(self, row, col):
    return self.data[row * self.cols + col]
  
  def add(self, other):
    assert self.rows == other.rows and self.cols == other.cols
    m = Matrix(self.rows, self.cols)
    for i in range(len(self.data)):
      m.data[i] = self.data[i] + other.data[i]
    return m

  def sub(self, other):
    assert self.rows == other.rows and self.cols == other.cols
    m = Matrix(self.rows, self.cols)
    for i in range(len(self.data)):
      m.data[i] = self.data[i] - other.data[i]
    return m

  def hadamard(self, other):
    assert self.rows == other.rows and self.cols == other.cols
    m = Matrix(self.rows, self.cols)
    for i in range(len(self.data)):
      m.data[i] = self.data[i] * other.data[i]
    return m

  def scalarmul(self, scalar):
    m = Matrix(self.rows, self.cols)
    for i in range(len(self.data)):
      m.data[i] = self.data[i] * scalar
    return m

  def matmul(self, other):
    assert self.cols == other.rows
    m = Matrix(self.rows, other.cols)

    for r in range(m.rows):
      for c in range(m.cols):
        for k in range(self.cols):
          m.data[r * m.cols + c] += self.get(r, k) * other.get(k, c)
    return m
  
  def transpose(self):
    m = Matrix(self.cols, self.rows)
    for r in range(m.rows):
      for c in range(m.cols):
          m.data[r * m.cols + c] = self.get(c, r)
    return m
        



