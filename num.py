import operation

class Number:
    
  def __init__(self, value, op = None):
    self.value = value
    self.op = op if op != None else operation.ConstantOp(self)
  
  def forward_derivative(self, var):
    return self.op.forward_derivative(var)

  def __add__(self, other):
    if isinstance(other, Number):
      return Number(self.value + other.value, operation.AddOp(self, other))
    elif isinstance(other, float) or isinstance(other, int):
      return Number(self.value + other, operation.AddOp(self, Number(other)))
    else:
      return NotImplemented

  def __radd__(self, other):
    if isinstance(other, Number):
      return Number(other.value + self.value, operation.AddOp(other, self))
    elif isinstance(other, float) or isinstance(other, int):
      return Number(other + self.value, operation.AddOp(Number(other), self))
    else:
      return NotImplemented

  def __sub__(self, other):
    if isinstance(other, Number):
      return Number(self.value - other.value, operation.SubOp(self, other))
    elif isinstance(other, float) or isinstance(other, int):
      return Number(self.value - other, operation.SubOp(self, Number(other)))
    else:
      return NotImplemented

  def __rsub__(self, other):
    if isinstance(other, Number):
      return Number(other.value - self.value, operation.SubOp(other, self))
    elif isinstance(other, float) or isinstance(other, int):
      return Number(other - self.value, operation.SubOp(Number(other), self))
    else:
      return NotImplemented

  def __mul__(self, other):
    if isinstance(other, Number):
      return Number(self.value * other.value, operation.MulOp(self, other))
    elif isinstance(other, float) or isinstance(other, int):
      return Number(self.value * other, operation.MulOp(self, Number(other)))
    else:
      return NotImplemented

  def __rmul__(self, other):
    if isinstance(other, Number):
      return Number(other.value * self.value, operation.MulOp(other, self))
    elif isinstance(other, float) or isinstance(other, int):
      return Number(other * self.value, operation.MulOp(Number(other), self))
    else:
      return NotImplemented

  def __truediv__(self, other):
    if isinstance(other, Number):
      return Number(self.value / other.value, operation.DivOp(self, other))
    elif isinstance(other, float) or isinstance(other, int):
      return Number(self.value / other, operation.DivOp(self, Number(other)))
    else:
      return NotImplemented

  def __rtruediv__(self, other):
    if isinstance(other, Number):
      return Number(other.value / self.value, operation.DivOp(other, self))
    elif isinstance(other, float) or isinstance(other, int):
      return Number(other / self.value, operation.DivOp(Number(other), self))
    else:
      return NotImplemented

  def __pow__(self, other):
    if isinstance(other, Number):
      return Number(self.value ** other.value, operation.PowOp(self, other))
    elif isinstance(other, float) or isinstance(other, int):
      return Number(self.value ** other, operation.PowOp(self, Number(other)))
    else:
      return NotImplemented

  def __rpow__(self, other):
    if isinstance(other, Number):
      return Number(other ** self.value, operation.PowOp(other, self))
    elif isinstance(other, float) or isinstance(other, int):
      return Number(other ** self.value, operation.PowOp(Number(other), self))
    else:
      return NotImplemented

  def __neg__(self):
    return Number(-self.value, operation.NegOp(self))
  
  def __str__(self):
    return str(self.value) + 'n'

