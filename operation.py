
class AddOp:
    
  def __init__(self, lvalue, rvalue):
    self.lvalue = lvalue
    self.rvalue = rvalue

  def derivative_wrt(self, var):
    return self.lvalue.derivative_wrt(var) + self.rvalue.derivative_wrt(var)

  def __str__(self):
    return '(+ ' + str(self.lvalue.op) + ' ' + str(self.rvalue.op) + ')'

class SubOp:
    
  def __init__(self, lvalue, rvalue):
    self.lvalue = lvalue
    self.rvalue = rvalue

  def derivative_wrt(self, var):
    return self.lvalue.derivative_wrt(var) - self.rvalue.derivative_wrt(var)

  def __str__(self):
    return '(- ' + str(self.lvalue.op) + ' ' + str(self.rvalue.op) + ')'

class MulOp:
    
  def __init__(self, lvalue, rvalue):
    self.lvalue = lvalue
    self.rvalue = rvalue

  def derivative_wrt(self, var):
    return self.lvalue.derivative_wrt(var) * self.rvalue + self.rvalue.derivative_wrt(var) * self.lvalue

  def __str__(self):
    return '(* ' + str(self.lvalue.op) + ' ' + str(self.rvalue.op) + ')'

class DivOp:
    
  def __init__(self, lvalue, rvalue):
    self.lvalue = lvalue
    self.rvalue = rvalue

  def derivative_wrt(self, var):
    return (self.lvalue.derivative_wrt(var) * self.rvalue - self.lvalue * self.rvalue.derivative_wrt(var)) / self.rvalue ** 2

  def __str__(self):
    return '(/ ' + str(self.lvalue.op) + ' ' + str(self.rvalue.op) + ')'

class PowOp:
    
  def __init__(self, lvalue, rvalue):
    self.lvalue = lvalue
    self.rvalue = rvalue

  def derivative_wrt(self, var):
    return self.rvalue * self.lvalue ** (self.rvalue - 1) * self.lvalue.derivative_wrt(var)

  def __str__(self):
    return '(^ ' + str(self.lvalue.op) + ' ' + str(self.rvalue.op) + ')'

class NegOp:
    
  def __init__(self, value):
    self.value = value

  def derivative_wrt(self, var):
    return -self.value.derivative_wrt(var)

  def __str__(self):
    return '(- ' + str(self.value.op) + ')'

class ConstantOp:
    
  def __init__(self, value):
    self.value = value

  def derivative_wrt(self, var):
    return 1 if var == self.value else 0

  def __str__(self):
    return str(self.value)
