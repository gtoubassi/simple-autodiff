from num import Number

num_tests = 0
num_passed = 0

def test_assert(cond, message=None):
  global num_tests, num_passed

  num_tests += 1
  if cond:
    num_passed += 1
  elif message is not None:
      print("Failure:", message)

def test_simple():
  test_assert(3.0 == (Number(1) + Number(2)).value, "1 + 2 == 3")
  test_assert(3.0 == (1 + Number(2)).value, "1 + 2 == 3")
  test_assert(3.0 == (Number(1) + 2).value, "1 + 2 == 3")

  test_assert(3.5 == (Number(.5) * Number(7)).value, ".5 * 7 = 3.5")
  test_assert(3.5 == (.5 * Number(7)).value, ".5 * 7 = 3.5")
  test_assert(3.5 == (Number(.5) * 7).value, ".5 * 7 = 3.5")

  test_assert(-1 == (Number(1) - Number(2)).value, "1 - 2 == -1")
  test_assert(-1 == (1 - Number(2)).value, "1 - 2 == -1")
  test_assert(-1 == (Number(1) - 2).value, "1 - 2 == -1")

  test_assert(3.4 == (Number(17) / Number(5)).value, "17 / 5 == 3.4")
  test_assert(3.4 == (17 / Number(5)).value, "17 / 5 == 3.4")
  test_assert(3.4 == (Number(17) / 5).value, "17 / 5 == 3.4")

  test_assert(81 == (Number(3) ** Number(4)).value, "3^4 == 81")
  test_assert(81 == (3 ** Number(4)).value, "3^4 == 81")
  test_assert(81 == (Number(3) ** 4).value, "3^4 == 81")

  test_assert(-17 == (-Number(17)).value, "-17")

def test_op_log():
  test_assert('(+ 1n 2n)' == str((Number(1) + Number(2)).op))
  test_assert('(+ 1n 2n)' == str((1 + Number(2)).op))
  test_assert('(+ 1n 2n)' == str((Number(1) + 2).op))

  test_assert('(- 1n 2n)' == str((Number(1) - Number(2)).op))
  test_assert('(- 1n 2n)' == str((1 - Number(2)).op))
  test_assert('(- 1n 2n)' == str((Number(1) - 2).op))

  test_assert('(* 2n 3n)' == str((Number(2) * Number(3)).op))
  test_assert('(* 2n 3n)' == str((2 * Number(3)).op))
  test_assert('(* 2n 3n)' == str((Number(2) * 3).op))

  test_assert('(/ 18n 5n)' == str((Number(18) / Number(5)).op))
  test_assert('(/ 18n 5n)' == str((18 / Number(5)).op))
  test_assert('(/ 18n 5n)' == str((Number(18) / 5).op))

  test_assert('(^ 3n 4n)' == str((Number(3) ** Number(4)).op))
  test_assert('(^ 3n 4n)' == str((3 ** Number(4)).op))
  test_assert('(^ 3n 4n)' == str((Number(3) ** 4).op))

  test_assert('(- 3n)' == str((-Number(3)).op))

  test_assert('(* (+ 1n 2n) 3n)' == str(((Number(1) + Number(2)) * Number(3)).op))
  test_assert('(- (- (/ (* (+ 1n 2n) 3n) 4n) 5n))' == str((-((Number(1) + Number(2)) * Number(3) / Number(4) - Number(5))).op))

def func_constant(x):
  return 3

def func_linear(x):
  return 3*x + 2

def func_add(x):
  return 3*x + x

def func_sub(x):
  return 3*x - x

def func_pow(x):
  return 3*x**2 + 2

def func_mul(x):
  return x*x

def func_div(x):
  return 3/(x + 10)

def func_neg(x):
  return -x

def func_add_chain(x):
  return 3*(x*x + 1/(x+10)) + (x*x + 1/(x+10))

def func_sub_chain(x):
  return 3*(x*x + 1/(x+10)) - (x*x + 1/(x+10))

def func_pow_chain(x):
  return 3*(x*x + 1/(x+10))**2 + 2

def func_mul_chain(x):
  return (x*x + 1/(x+10))*(x*x + 1/(x+10))

def func_div_chain(x):
  return 3/((x*x + 1/(x+10)) + 10)

def func_neg_chain(x):
  return -(x*x + 1/(x+10))

def test_derivative(func, low_range, high_range):
  for i in range(10):
    x = low_range + i * (high_range - low_range) / 10
    eval_native = func(x)
    number_x = Number(x)
    eval_number = func(number_x)
    eval_number = eval_number if isinstance(eval_number, Number) else Number(eval_number)
    test_assert(eval_native == eval_number.value)
    
    # Finite difference
    finite_diff_derivative = (func(x + 1e-7) - eval_native) / 1e-7
    
    # Autodiff fwd mode
    autodiff_fwd = eval_number.forward_derivative(number_x)
    test_assert(abs(finite_diff_derivative - autodiff_fwd.value) < 1e-3)
    
    # Autodiff fwd mode
    autodiff_fwd = eval_number.bwd(number_x)
    test_assert(abs(finite_diff_derivative - autodiff_fwd) < 1e-3)
    
def test_simple_derivative():
  test_derivative(func_constant, -2.0, 2.0)
  test_derivative(func_add, -2.0, 2.0)
  test_derivative(func_sub, -2.0, 2.0)
  test_derivative(func_pow, -2.0, 2.0)
  test_derivative(func_mul, -2.0, 2.0)
  test_derivative(func_div, -2.0, 2.0)
  test_derivative(func_neg, -2.0, 2.0)
  test_derivative(func_linear, -2.0, 2.0)
  test_derivative(func_add_chain, -2.0, 2.0)
  test_derivative(func_sub_chain, -2.0, 2.0)
  test_derivative(func_pow_chain, -2.0, 2.0)
  test_derivative(func_mul_chain, -2.0, 2.0)
  test_derivative(func_div_chain, -2.0, 2.0)
  test_derivative(func_neg_chain, -2.0, 2.0)
  
def main():
  global num_tests, num_passed

  test_simple()
  test_op_log()
  test_simple_derivative()
  if num_tests > num_passed:
    print("%d FAILED!" % (num_tests - num_passed))
  print("%d / %d passed" % (num_passed, num_tests))
  
if __name__ == "__main__":
  main()

