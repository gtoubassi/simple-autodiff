
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
    
  def copy(self):
    m = Matrix(self.rows, self.cols)
    for i in range(len(self.data)):
      m.data[i] = self.data[i]
    return m

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
      assert self.cols == 1, "Must specify 2 arguments for indexing a non column matrix (%d, %d)" % (self.rows, self.cols)
      return self.get(key, 0)
    if isinstance(key, tuple):
      assert len(key) == 2, "Must specify 2 arguments for indexing a matrix (%d, %d)" % (self.rows, self.cols)
      return self.get(key[0], key[1])
    
    raise TypeError

  def __setitem__(self, key, value):
    if isinstance(key, int):
      assert self.cols == 1, "Must specify 2 arguments for indexing a non column matrix (%d, %d)" % (self.rows, self.cols)
      self.data[key * self.cols + 0] = value
    elif isinstance(key, tuple):
      assert len(key) == 2, "Must specify 2 arguments for indexing a matrix (%d, %d)" % (self.rows, self.cols)
      self.data[key[0] * self.cols + key[1]] = value
    else:
      raise TypeError

  def __add__(self, other):
    assert self.rows == other.rows and self.cols == other.cols
    m = Matrix(self.rows, self.cols)
    for i in range(len(self.data)):
      m.data[i] = self.data[i] + other.data[i]
    return m

  def __sub__(self, other):
    assert self.rows == other.rows and self.cols == other.cols
    m = Matrix(self.rows, self.cols)
    for i in range(len(self.data)):
      m.data[i] = self.data[i] - other.data[i]
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

