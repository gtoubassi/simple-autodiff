
class AddOp:
    
  def __init__(self, lvalue, rvalue):
    self.lvalue = lvalue
    self.rvalue = rvalue

  def forward_derivative(self, var):
    return self.lvalue.forward_derivative(var) + self.rvalue.forward_derivative(var)

  def __str__(self):
    return '(+ ' + str(self.lvalue.op) + ' ' + str(self.rvalue.op) + ')'

class SubOp:
    
  def __init__(self, lvalue, rvalue):
    self.lvalue = lvalue
    self.rvalue = rvalue

  def forward_derivative(self, var):
    return self.lvalue.forward_derivative(var) - self.rvalue.forward_derivative(var)

  def __str__(self):
    return '(- ' + str(self.lvalue.op) + ' ' + str(self.rvalue.op) + ')'

class MulOp:
    
  def __init__(self, lvalue, rvalue):
    self.lvalue = lvalue
    self.rvalue = rvalue

  def forward_derivative(self, var):
    return self.lvalue.forward_derivative(var) * self.rvalue + self.rvalue.forward_derivative(var) * self.lvalue

  def __str__(self):
    return '(* ' + str(self.lvalue.op) + ' ' + str(self.rvalue.op) + ')'

class DivOp:
    
  def __init__(self, lvalue, rvalue):
    self.lvalue = lvalue
    self.rvalue = rvalue

  def forward_derivative(self, var):
    return (self.lvalue.forward_derivative(var) * self.rvalue - self.lvalue * self.rvalue.forward_derivative(var)) / self.rvalue ** 2

  def __str__(self):
    return '(/ ' + str(self.lvalue.op) + ' ' + str(self.rvalue.op) + ')'

class PowOp:
    
  def __init__(self, lvalue, rvalue):
    self.lvalue = lvalue
    self.rvalue = rvalue

  def forward_derivative(self, var):
    return self.rvalue * self.lvalue ** (self.rvalue - 1) * self.lvalue.forward_derivative(var)

  def __str__(self):
    return '(^ ' + str(self.lvalue.op) + ' ' + str(self.rvalue.op) + ')'

class NegOp:
    
  def __init__(self, value):
    self.value = value

  def forward_derivative(self, var):
    return -self.value.forward_derivative(var)

  def __str__(self):
    return '(- ' + str(self.value.op) + ')'

class ConstantOp:
    
  def __init__(self, value):
    self.value = value

  def forward_derivative(self, var):
    return 1 if var == self.value else 0

  def __str__(self):
    return str(self.value)
