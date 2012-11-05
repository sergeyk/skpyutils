from context import *
from skpyutils.table import Table

import operator

def filtering_test():
  arr = np.array([
    [0.2, 0.3, 0.4],
    [0.1, 0.3, 0.6],
    [0.6, 0.3, 0.6]])
  cols = ['a','b','c']
  index = ['one','two','three']
  name = 'test_table'
  t = Table(arr, cols, index, name)

  # filter by equality
  t2 = t.filter_on_column('c',0.6)
  arr2 = np.array([
    [0.1, 0.3, 0.6],
    [0.6, 0.3, 0.6]])
  cols2 = ['a','b','c']
  index2 = ['two','three']
  name2 = 'test_table'
  assert(t2.shape == arr2.shape and np.all(t2.arr == arr2) and t2.cols == cols2 and t2.index == index2 and t2.name == name2)

  # filter by < 0.6
  t2 = t.filter_on_column('c',0.6,operator.lt)
  arr2 = np.array([
    [0.2, 0.3, 0.4]])
  cols2 = ['a','b','c']
  index2 = ['one']
  name2 = 'test_table'
  assert(t2.shape == arr2.shape and np.all(t2.arr == arr2) and t2.cols == cols2 and t2.index == index2 and t2.name == name2)
  
  # filter by < 0.6 and omit column
  t2 = t.filter_on_column('c',0.6,operator.lt,omit=True)
  arr2 = np.array([
    [0.2, 0.3]])
  cols2 = ['a','b']
  index2 = ['one']
  name2 = 'test_table'
  assert(t2.shape == arr2.shape and np.all(t2.arr == arr2) and t2.cols == cols2 and t2.index == index2 and t2.name == name2)

def overall_and_subset_test():
  arr = np.array([
    [0.2, 0.3, 0.4],
    [0.1, 0.3, 0.6],
    [0.6, 0.3, 0.6]])
  cols = ['a','b','c']
  index = ['one','two','three']
  name = 'test_table'
  t = Table(arr, cols, index, name)

  # test init
  assert(t.shape == arr.shape and np.all(t.arr == arr) and t.cols == cols and t.index == index and t.name == name)

  # test copy
  t2 = t.copy()
  assert(t2.shape == arr.shape and np.all(t2.arr == arr) and t2.cols == cols and t2.index == index and t2.name == name)

  # change values in the copy and make sure original is unmodified
  t2.arr[0,:] = 1
  t2.index[0] = 'one_m'
  t2.cols[0] = 'big A'
  t2.name = 'modified_table'
  assert(t.shape == arr.shape and np.all(t.arr == arr) and t.cols == cols and t.index == index and t.name == name)

  ## subset the first and last rows of t2
  arr2 = np.array([
    [1, 1, 1],
    [0.6, 0.3, 0.6]])
  index2 = ['one_m','three']
  cols2 = ['big A','b','c']
  name2 = 'modified_table'

  # by list of indices
  t3 = t2.row_subset([0,2])
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.row_subset_arr([0,2])
  assert(np.all(t3_arr==arr2))
  # by list of names
  t3 = t2.row_subset(['one_m','three'])
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.row_subset_arr(['one_m','three'])
  assert(np.all(t3_arr==arr2))

  # by array of indices
  t3 = t2.row_subset(np.array([0,2]))
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.row_subset_arr(np.array([0,2]))
  assert(np.all(t3_arr==arr2))
  # by array of names
  t3 = t2.row_subset(np.array(['one_m','three']))
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.row_subset_arr(['one_m','three'])
  assert(np.all(t3_arr==arr2))

  # subset just the middle row of t2
  arr2 = np.array([
    [0.1, 0.3, 0.6]])
  cols2 = ['big A','b','c']
  index2 = ['two']
  name2 = 'modified_table'
  # by index
  t3 = t2.row_subset(1)
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.row_subset_arr(1)
  assert(np.all(t3_arr==arr2))
  # by name
  t3 = t2.row_subset('two')
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.row_subset_arr('two')
  assert(np.all(t3_arr==arr2.squeeze()))

  ## subset the first and last columns of t2
  arr2 = np.array([
    [1, 1],
    [0.1, 0.6],
    [0.6, 0.6]])
  index2 = ['one_m','two','three']
  cols2 = ['big A','c']
  name2 = 'modified_table'

  # by list of indices
  t3 = t2.subset([0,2])
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.subset_arr([0,2])
  assert(np.all(t3_arr==arr2))
  # by list of names
  t3 = t2.subset(['big A','c'])
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.subset_arr(['big A','c'])
  assert(np.all(t3_arr==arr2))

  # by array of indices
  t3 = t2.subset(np.array([0,2]))
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.subset_arr(np.array([0,2]))
  assert(np.all(t3_arr==arr2))
  # by array of names
  t3 = t2.subset(np.array(['big A','c']))
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.subset_arr(np.array(['big A','c']))
  assert(np.all(t3_arr==arr2))

  ## subset just the middle column of t2
  arr2 = np.array([
    [1],
    [0.3],
    [0.3]])
  index2 = ['one_m','two','three']
  cols2 = ['b']
  name2 = 'modified_table'
  # by index
  t3 = t2.subset(1)
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.subset_arr(np.array(['b']))
  assert(np.all(t3_arr==arr2.squeeze()))
  # by name
  t3 = t2.subset('b')
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.subset_arr(['b'])
  assert(np.all(t3_arr==arr2.squeeze()))

  ## row_subset without index
  t2 = Table(arr, cols, None, name)
  # subset the first and last rows of t2
  arr2 = np.array([
    [0.2, 0.3, 0.4],
    [0.6, 0.3, 0.6]])
  index2 = None
  cols2 = ['a','b','c']
  name2 = 'test_table'

  # by list of indices
  t3 = t2.row_subset([0,2])
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.row_subset_arr([0,2])
  assert(np.all(t3_arr==arr2))

  # by array of indices
  t3 = t2.row_subset(np.array([0,2]))
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.row_subset_arr(np.array([0,2]))
  assert(np.all(t3_arr==arr2))

  ## subset just the middle row of t2
  arr2 = np.array([
    [0.1, 0.3, 0.6]])
  cols2 = ['a','b','c']
  name2 = 'test_table'
  # by index
  t3 = t2.row_subset(1)
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.row_subset_arr(1)
  assert(np.all(t3_arr==arr2.squeeze()))

  ## subset just the middle value of t2
  arr2 = np.array([
    [0.3]])
  cols2 = ['b']
  name2 = 'test_table'
  # by index
  t3 = t2.row_subset(1).subset(1)
  assert(t3.shape == arr2.shape and np.all(t3.arr == arr2) and t3.cols == cols2 and t3.index == index2 and t3.name == name2)
  t3_arr = t2.row_subset(1).subset_arr(1)
  assert(np.all(t3_arr==np.array([0.3])))
