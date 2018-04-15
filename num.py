# TODO
#
# 1. Move auto diff code into autodiff.py
# 2. Handle jacobians as well
# 3. Centralize finite diff code (into autodiff?)
# 4. Support for exp(Number) which means fixing pow's autodiff.  The full monty is
#   https://math.stackexchange.com/questions/210915/derivative-of-a-function-raised-to-the-power-of-x

class Number:
  opcount = 0

  def __init__(self, value):
    self.value = value
    self.parents = []
    self.children = []
    self.grad_value = None
  
  def forward_autodiff(self, var):
    Number.opcount = 0
    self._reset_grad()
    var._reset_grad()
    var.grad_value = 1
    z = self._do_forward_autodiff()
    return z
  
  def reverse_autodiff(self, var):
    Number.opcount = 0
    self._reset_grad()
    var._reset_grad()
    self.grad_value = 1
    z = var._do_reverse_autodiff()
    return z
  
  def _reset_grad(self):
    self.grad_value = None
    for partial, parent in self.parents:
      parent._reset_grad()

  def _do_forward_autodiff(self):
    # Strictly speaking forward pass doesn't need caching since
    # we are executing in a non repetitive order
    if self.grad_value is None:
      Number.opcount += 1
      self.grad_value = 0
      for partial, parent in self.parents:
        self.grad_value += partial * parent._do_forward_autodiff()
    
    return self.grad_value
    
  def _do_reverse_autodiff(self):
    if self.grad_value is None:
      Number.opcount += 1
      self.grad_value = 0
      for partial, child in self.children:
        self.grad_value += partial * child._do_reverse_autodiff()
  
    return self.grad_value
    
  def add(self, other):
    z = Number(self.value + other.value)
    z.parents.append((1.0, self))
    z.parents.append((1.0, other))
    self.children.append((1.0, z))
    other.children.append((1.0, z))
    return z

  def sub(self, other):
    z = Number(self.value - other.value)
    z.parents.append((1.0, self))
    z.parents.append((-1.0, other))
    self.children.append((1.0, z))
    other.children.append((-1.0, z))
    return z

  def mul(self, other):
    z = Number(self.value * other.value)
    z.parents.append((other.value, self))
    z.parents.append((self.value, other))
    self.children.append((other.value, z))
    other.children.append((self.value, z))
    return z
    
  def truediv(self, other):
    z = Number(self.value / other.value)
    partial_z_wrt_self = 1 / other.value
    partial_z_wrt_other = -self.value / other.value**2
    z.parents.append((partial_z_wrt_self, self))
    z.parents.append((partial_z_wrt_other, other))
    self.children.append((partial_z_wrt_self, z))
    other.children.append((partial_z_wrt_other, z))
    return z
    
  def pow(self, other):
    z = Number(self.value ** other.value)
    partial_z_wrt_self = other.value * self.value ** (other.value - 1)
    z.parents.append((partial_z_wrt_self, self))
    self.children.append((partial_z_wrt_self, z))
    return z
    
  def neg(self):
    z = Number(-self.value)
    z.parents.append((-1.0, self))
    self.children.append((-1, z))
    return z
    
  def abs(self):
    return self if self.value >= 0 else self.neg()
    
  def __add__(self, other):
    other = other if isinstance(other, Number) else Number(other)
    return self.add(other)

  def __radd__(self, other):
    other = other if isinstance(other, Number) else Number(other)
    return other.add(self)

  def __sub__(self, other):
    other = other if isinstance(other, Number) else Number(other)
    return self.sub(other)

  def __rsub__(self, other):
    other = other if isinstance(other, Number) else Number(other)
    return other.sub(self)

  def __mul__(self, other):
    other = other if isinstance(other, Number) else Number(other)
    return self.mul(other)

  def __rmul__(self, other):
    other = other if isinstance(other, Number) else Number(other)
    return other.mul(self)

  def __truediv__(self, other):
    other = other if isinstance(other, Number) else Number(other)
    return self.truediv(other)

  def __rtruediv__(self, other):
    other = other if isinstance(other, Number) else Number(other)
    return other.truediv(self)

  def __pow__(self, other):
    other = other if isinstance(other, Number) else Number(other)
    return self.pow(other)

  def __rpow__(self, other):
    other = other if isinstance(other, Number) else Number(other)
    return other.pow(self)

  def __neg__(self):
    return self.neg()
  
  def __abs__(self):
    return self.abs()

  def __eq__(self, other):
    return self.value == other.value
  
  def __lt__(self, other):
    other = other.value if isinstance(other, Number) else other
    return self.value < other
      
  def __le__(self, other):
    other = other.value if isinstance(other, Number) else other
    return self.value <= other
  
  def __gt__(self, other):
    other = other.value if isinstance(other, Number) else other
    return self.value > other
      
  def __ge__(self, other):
    other = other.value if isinstance(other, Number) else other
    return self.value >= other
      
  def __str__(self):
    return str(self.value) + 'n'
