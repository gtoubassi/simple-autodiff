import operation

class Number:
    
  def __init__(self, value, op = None):
    self.value = value
    self.op = op if op != None else operation.ConstantOp(self)
    self.parents = []
    self.children = []
    self.grad_value = None
  
  def forward_derivative(self, var):
    return self.op.forward_derivative(var)
  
  def fwd(self, var):
    var.grad_value = 1
    return self.dofwd()
  
  def bwd(self, var):
    self.grad_value = 1
    return var.dobwd()
  
  def dofwd(self):
    if self.grad_value is None:
      self.grad_value = 0
      for parent in self.parents:
        self.grad_value += parent[0] * parent[1].dofwd()
    
    return self.grad_value
    
  def dobwd(self):
    if self.grad_value is None:
      self.grad_value = 0
      for child in self.children:
        self.grad_value += child[0] * child[1].dobwd()
    
    return self.grad_value
    
  def add(self, other):
    z = Number(self.value + other.value, operation.AddOp(self, other))
    z.parents.append((1.0, self))
    z.parents.append((1.0, other))
    self.children.append((1.0, z))
    other.children.append((1.0, z))
    return z

  def sub(self, other):
    z = Number(self.value - other.value, operation.SubOp(self, other))
    z.parents.append((1.0, self))
    z.parents.append((-1.0, other))
    self.children.append((1.0, z))
    other.children.append((-1.0, z))
    return z

  def mul(self, other):
    z = Number(self.value * other.value, operation.MulOp(self, other))
    z.parents.append((other.value, self))
    z.parents.append((self.value, other))
    self.children.append((other.value, z))
    other.children.append((self.value, z))
    return z
    
  def truediv(self, other):
    z = Number(self.value / other.value, operation.DivOp(self, other))
    partial_z_wrt_self = 1 / other.value
    partial_z_wrt_other = -self.value / other.value**2
    z.parents.append((partial_z_wrt_self, self))
    z.parents.append((partial_z_wrt_other, other))
    self.children.append((partial_z_wrt_self, z))
    other.children.append((partial_z_wrt_other, z))
    return z
    
  def pow(self, other):
    z = Number(self.value ** other.value, operation.PowOp(self, other))
    partial_z_wrt_self = other.value * self.value ** (other.value - 1)
    z.parents.append((partial_z_wrt_self, self))
    self.children.append((partial_z_wrt_self, z))
    return z
    
  def neg(self):
    z = Number(-self.value, operation.NegOp(self))
    z.parents.append((-1.0, self))
    self.children.append((-1, z))
    return z
    
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
  
  def __str__(self):
    return str(self.value) + 'n'

