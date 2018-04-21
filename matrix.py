from scalar import Scalar
import math

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
    elif isinstance(initializer, Scalar):
      # For now just allow wrapping in a 1x1, we need to decide copy semantics.
      assert rows == 1 and cols == 1
      self.data[0] = initializer
    
  def copy(self):
    m = Matrix(self.rows, self.cols)
    for i in range(len(self.data)):
      m.data[i] = self.data[i]
    return m

  def reshape_inplace(self, new_rows, new_cols):
    assert self.rows * self.cols == new_rows * new_cols
    self.rows = new_rows
    self.cols = new_cols

  def _apply(self, func):
    for i in range(len(self.data)):
      func(self.data[i])
  
  def apply_copy(self, func):
    m = Matrix(self.rows, self.cols)
    for i in range(len(self.data)):
      m.data[i] = func(self.data[i])
    return m
  
  def compare(self, other, tolerance = 0.0):
    if self.rows != other.rows or self.cols != other.cols:
      return False
    for i in range(len(self.data)):
      if abs(self.data[i] - other.data[i]) > tolerance:
        return False
    return True
  
  def euclidean_norm(self):
    total = 0
    for i in range(len(self.data)):
      total += self.data[i]*self.data[i]
    return math.sqrt(total)

  def gather_rows(self, indices):
    m = Matrix(len(indices), self.cols)
    for i in range(len(indices)):
      for col in range(self.cols):
        m[i, col] = self[indices[i], col]
    return m

  def get(self, row, col):
    return self.data[row * self.cols + col]
  
  def matmul(self, other):
    assert self.cols == other.rows
    m = Matrix(self.rows, other.cols)

    for r in range(m.rows):
      for c in range(m.cols):
        for k in range(self.cols):
          m.data[r * m.cols + c] += self.get(r, k) * other.get(k, c)
    return m

  def reduce_sum(self):
    s = 0
    for i in range(len(self.data)):
      s += self.data[i]
    return s
  
  def transpose(self):
    m = Matrix(self.cols, self.rows)
    for r in range(m.rows):
      for c in range(m.cols):
          m.data[r * m.cols + c] = self.get(c, r)
    return m
  
  def __getitem__(self, key):
    if isinstance(key, int):
      if self.cols == 1:
        return self.get(key, 0)
      else:
        m = Matrix(1, self.cols)
        for i in range(self.cols):
          m[0, i] = self.get(key, i)
        return m
    if isinstance(key, tuple):
      assert len(key) == 2, "Must specify 2 arguments for indexing a matrix (%d, %d)" % (self.rows, self.cols)
      return self.get(key[0], key[1])
    
    raise TypeError

  def __setitem__(self, key, value):
    if isinstance(key, int):
      if isinstance(value, Matrix):
        assert self.cols == value.cols and value.rows == 1
        for c in range(value.cols):
          self.data[key * self.cols + c] = value[0, c]
      else:
        assert self.cols == 1, "Must specify 2 arguments for indexing a non column matrix (%d, %d)" % (self.rows, self.cols)
        self.data[key * self.cols + 0] = value
    elif isinstance(key, tuple):
      assert len(key) == 2, "Must specify 2 arguments for indexing a matrix (%d, %d)" % (self.rows, self.cols)
      self.data[key[0] * self.cols + key[1]] = value
    else:
      raise TypeError

  def __add__(self, other):
    if isinstance(other, Matrix):
      assert self.rows == other.rows and self.cols == other.cols
      m = Matrix(self.rows, self.cols)
      for i in range(len(self.data)):
        m.data[i] = self.data[i] + other.data[i]
      return m
    elif isinstance(other, float) or isinstance(other, int) or isinstance(other, Scalar):
      m = Matrix(self.rows, self.cols)
      for i in range(len(self.data)):
        m.data[i] = self.data[i] + other
      return m

  def __sub__(self, other):
    if isinstance(other, Matrix):
      assert self.rows == other.rows and self.cols == other.cols
      m = Matrix(self.rows, self.cols)
      for i in range(len(self.data)):
        m.data[i] = self.data[i] - other.data[i]
      return m
    elif isinstance(other, float) or isinstance(other, int) or isinstance(other, Scalar):
      m = Matrix(self.rows, self.cols)
      for i in range(len(self.data)):
        m.data[i] = self.data[i] - other
      return m

  def __mul__(self, other):
    m = Matrix(self.rows, self.cols)
    if isinstance(other, Matrix):
      assert self.rows == other.rows and self.cols == other.cols
      for i in range(len(self.data)):
        m.data[i] = self.data[i] * other.data[i]
      return m
    elif isinstance(other, float) or isinstance(other, int):
      for i in range(len(self.data)):
        m.data[i] = self.data[i] * other
      return m

  def __rmul__(self, other):
    return self.__mul__(other)

  def __pow__(self, other):
    if isinstance(other, float) or isinstance(other, int):
      m = Matrix(self.rows, self.cols)
      for i in range(len(self.data)):
        m.data[i] = self.data[i] ** other
      return m

  def __truediv__(self, other):
    m = Matrix(self.rows, self.cols)
    if isinstance(other, Matrix):
      assert self.rows == other.rows and self.cols == other.cols
      for i in range(len(self.data)):
        m.data[i] = self.data[i] / other.data[i]
      return m
    elif isinstance(other, float) or isinstance(other, int):
      for i in range(len(self.data)):
        m.data[i] = self.data[i] / other
      return m

def convert_to_scalar(m):
  if isinstance(m, Scalar):
    return m
  if isinstance(m, Matrix):
    return Matrix(m.rows, m.cols, lambda r, c: Scalar(m[r, c]))
  if isinstance(m, float) or isinstance(m, int):
    return Scalar(m)
  return m

def convert_from_scalar(m):
  if isinstance(m, Scalar):
    return m.value
  if isinstance(m, Matrix):
    return Matrix(m.rows, m.cols, lambda r, c: m[r,c].value if isinstance(m[r,c], Scalar) else m[r,c])
  return m

