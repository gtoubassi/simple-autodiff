from matrix import Matrix
from number import Number

num_tests = 0
num_passed = 0

def test_assert(cond, message=None):
  global num_tests, num_passed

  num_tests += 1
  assert(cond)
    
  if cond:
    num_passed += 1
  elif message is not None:
    print("Failure:", message)

def test_initialization():
  m = Matrix(3, 2)
  test_assert(m.get(2,1) == 0.0)
  test_assert(m.get(1,0) == 0.0)
  m2 = Matrix(3, 2, [[0, 0], [0, 0], [0, 0]])
  test_assert(m2.get(2,1) == 0.0)
  test_assert(m2.get(1,0) == 0.0)

  test_assert(m.compare(m2))
  m2 = Matrix(3, 2, [[1, 0], [0, 0], [0, 0]])
  test_assert(not m.compare(m2))
  
  m3 = Matrix(3, 2, lambda r, c: r + 2*c)
  m4 = Matrix(3, 2, [[0, 2], [1, 3], [2, 4]])
  test_assert(m3.compare(m4))
  test_assert(m4.compare(m3))

def test_add():
  m1 = Matrix(3, 2, [[1, 2], [3, 4], [5, 6]])
  m2 = Matrix(3, 2, [[7, 8], [9, 10], [11, 12]])
  m1plusm2 = m1.add(m2)
  m1plusm2_correct = Matrix(3, 2, [[8, 10], [12, 14], [16, 18]])
  test_assert(m1plusm2.compare(m1plusm2_correct))

def test_sub():
  m1 = Matrix(3, 2, [[1, 2], [3, 4], [5, 6]])
  m2 = Matrix(3, 2, [[7, 9], [11, 13], [15, 17]])
  m1subm2 = m1.sub(m2)
  m1subm2_correct = Matrix(3, 2, [[-6, -7], [-8, -9], [-10, -11]])
  test_assert(m1subm2.compare(m1subm2_correct))

def test_hadamard():
  m1 = Matrix(3, 2, [[1, 2], [3, 4], [5, 6]])
  m2 = Matrix(3, 2, [[7, 9], [11, 13], [15, 17]])
  m1hadamardm2 = m1.hadamard(m2)
  m1hadamardm2_correct = Matrix(3, 2, [[7, 18], [33, 52], [75, 102]])
  test_assert(m1hadamardm2.compare(m1hadamardm2_correct))

def test_scalarmul():
  m = Matrix(3, 2, [[1, 2], [3, 4], [5, 6]])
  m = m.scalarmul(.5)
  correct = Matrix(3, 2, [[.5, 1], [1.5, 2], [2.5, 3]])
  test_assert(m.compare(correct))

def test_matmul():
  m1 = Matrix(3, 2, [[1, 2], [3, 4], [5, 6]])
  m2 = Matrix(2, 3, [[7, 8, 9], [10, 11, 12]])
  m1mulm2 = m1.matmul(m2)
  m1mulm2_correct = Matrix(3, 3, [[1*7 + 2*10, 1*8 + 2*11, 1*9 + 2*12], [3*7 + 4*10, 3*8 + 4*11, 3*9 + 4*12], [5*7 + 6*10, 5*8 + 6*11, 5*9 + 6*12]])
  test_assert(m1mulm2.compare(m1mulm2_correct))
  
  v1 = Matrix(3, 1, [[2], [4], [6]])
  v2 = Matrix(3, 1, [[.5], [2], [3]])
  dot = v1.transpose().matmul(v2)
  test_assert(dot.compare(Matrix(1,1,27)))

def test_transpose():
  m1 = Matrix(3, 2, [[1, 2], [3, 4], [5, 6]])
  m1transpose = m1.transpose()
  m1transpose_correct = Matrix(2, 3, [[1, 3, 5], [2, 4, 6]])
  test_assert(m1transpose.compare(m1transpose_correct))

def convert_to_number(m):
  return Matrix(m.rows, m.cols, lambda r, c: Number(m.get(r, c)))
  
def test_with_numbers():
  m1 = convert_to_number(Matrix(3, 2, [[1, 2], [3, 4], [5, 6]]))
  m2 = convert_to_number(Matrix(2, 3, [[7, 8, 9], [10, 11, 12]]))
  m1mulm2 = m1.matmul(m2)
  m1mulm2_correct = convert_to_number(Matrix(3, 3, [[1*7 + 2*10, 1*8 + 2*11, 1*9 + 2*12], [3*7 + 4*10, 3*8 + 4*11, 3*9 + 4*12], [5*7 + 6*10, 5*8 + 6*11, 5*9 + 6*12]]))
  test_assert(m1mulm2.compare(m1mulm2_correct))
  
def main():
  global num_tests, num_passed

  test_initialization()
  test_add()
  test_sub()
  test_hadamard()
  test_scalarmul()
  test_matmul()
  test_transpose()
  test_with_numbers()

  if num_tests > num_passed:
    print("%d FAILED!" % (num_tests - num_passed))
  print("%d / %d passed" % (num_passed, num_tests))
  
if __name__ == "__main__":
  main()

