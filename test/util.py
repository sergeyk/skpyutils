from context import *
from skpyutils import util

import itertools

class Basic(unittest.TestCase):
  def setUp(self):
    self.test_dir = os.path.dirname(__file__)

  def test_random_subset_up_to_N(self):
    self.assertRaises(ValueError, util.random_subset_up_to_N, -1, -1)
    self.assertRaises(ValueError, util.random_subset_up_to_N, -1, 1)
    self.assertRaises(ValueError, util.random_subset_up_to_N, 1, 0)

    x = util.random_subset_up_to_N(10,1)
    self.assertIs(type(x),np.ndarray)
    assert(len(x)==1)

    x = util.random_subset_up_to_N(10,2)
    self.assertIs(type(x),np.ndarray)
    assert(len(x)==2)
    assert(np.all(x<10))
    assert(np.all(x>=0))

    x = util.random_subset_up_to_N(100,100)
    # very unlikely that a random permutation will exactly equal
    assert(not np.all(x == np.arange(0,100)))
    # but sorted, it definitely should
    assert(np.all(sorted(x) == np.arange(0,100)))

    x = util.random_subset_up_to_N(100)
    assert(not np.all(x == np.arange(0,100)))
    assert(np.all(sorted(x) == np.arange(0,100)))

    x = util.random_subset_up_to_N(100,10000)
    assert(not np.all(x == np.arange(0,100)))
    assert(np.all(sorted(x) == np.arange(0,100)))

    # Some more tests for good measure
    Ns = [1,2,10,100]
    max_nums = [1,50]
    for N,max_num in itertools.product(Ns, max_nums):
      r = util.random_subset_up_to_N(N, max_num)
      assert(len(r)==min(N,max_num))
      assert(max(r)<=N)
      assert(min(r)>=0)

  def test_random_subset_up_to_N_exception(self):
    Ns = [-2, 0]
    max_nums = [1,50]
    for N,max_num in itertools.product(Ns, max_nums):
      assert_raises(ValueError, util.random_subset_up_to_N, N, max_num)
    Ns = [1, 10, 100]
    max_nums = [-4, 0]
    for N,max_num in itertools.product(Ns, max_nums):
      assert_raises(ValueError, util.random_subset_up_to_N, N, max_num)

  def test_random_subset(self):
    x = util.random_subset(np.arange(10),1)
    self.assertIs(type(x),list)

    x = util.random_subset(np.arange(10),2)
    self.assertIs(type(x),list)
    assert(len(x)==2)
    assert(np.all(np.array(x)<10))
    assert(np.all(np.array(x)>=0))

    x = util.random_subset(np.arange(100),200)
    assert(len(x)==100)
    assert(not np.all(x == np.arange(0,100)))
    assert(np.all(sorted(x) == np.arange(0,100)))

    x = util.random_subset(np.arange(100))
    assert(len(x)==100)
    assert(not np.all(x == np.arange(0,100)))
    assert(np.all(sorted(x) == np.arange(0,100)))

    # some more tests for good measure
    l = range(100,120)
    max_num = 10
    r = util.random_subset(l, max_num)
    assert(len(r)==min(len(l),max_num))
    assert(max(r)<=max(l))
    assert(min(r)>=min(l))

    l = np.array(range(0,120))
    max_num = 10
    r = util.random_subset(l, max_num)
    assert(len(r)==min(len(l),max_num))
    assert(max(r)<=max(l))
    assert(min(r)>=min(l))

  def test_determine_bin(self):
    values = np.array([0, 0.05,0.073,0.0234,0.1,0.13423,0.123534,0.1253,0.212,0.2252,0.43,0.3]).astype(float)
    bounds = np.array([0,0.1,0.2,0.3,np.max(values)])    
    bins = util.determine_bin(values, bounds, 4)
    bins_gt = np.array([0,0,0,0,1,1,1,1,2,2,3,3])  
    assert_equal(bins, bins_gt)

  def test_histogram(self):
    data = np.random.randint(0,10,(5000,)) 
    assert_almost_equal(util.histogram(data, 5), np.tile(1000, (1,5)),-2)  

if __name__ == '__main__':
  unittest.main()
  