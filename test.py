from num import Number

def test_simple():
  assert 3.0 == (Number(1) + Number(2)).value, "1 + 2 == 3"
  assert 3.0 == (1 + Number(2)).value, "1 + 2 == 3"
  assert 3.0 == (Number(1) + 2).value, "1 + 2 == 3"

  assert 3.5 == (Number(.5) * Number(7)).value, ".5 * 7 = 3.5"
  assert 3.5 == (.5 * Number(7)).value, ".5 * 7 = 3.5"
  assert 3.5 == (Number(.5) * 7).value, ".5 * 7 = 3.5"

  assert -1 == (Number(1) - Number(2)).value, "1 - 2 == -1"
  assert -1 == (1 - Number(2)).value, "1 - 2 == -1"
  assert -1 == (Number(1) - 2).value, "1 - 2 == -1"

  assert 3.4 == (Number(17) / Number(5)).value, "17 / 5 == 3.4"
  assert 3.4 == (17 / Number(5)).value, "17 / 5 == 3.4"
  assert 3.4 == (Number(17) / 5).value, "17 / 5 == 3.4"

  assert 81 == (Number(3) ** Number(4)).value, "3^4 == 81"
  assert 81 == (3 ** Number(4)).value, "3^4 == 81"
  assert 81 == (Number(3) ** 4).value, "3^4 == 81"

  assert -17 == (-Number(17)).value, "-17"

def test_op():
  assert '(+ 1n 2n)' == str((Number(1) + Number(2)).op)
  assert '(* (+ 1n 2n) 3n)' == str(((Number(1) + Number(2)) * Number(3)).op)
  assert '(- (- (/ (* (+ 1n 2n) 3n) 4n) 5n))' == str((-((Number(1) + Number(2)) * Number(3) / Number(4) - Number(5))).op)

def simple_line_func(x):
  return 2*x**2 + 3

def test_simple_derivative():
  x = Number(5)
  y = simple_line_func(x)
  dy_dx = y.forward_derivative(x)
  print(dy_dx.op)
  print(dy_dx)
  
def main():
  test_simple()
  test_op()
  test_simple_derivative()
  
if __name__ == "__main__":
  main()

