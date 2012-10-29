from context import *
from skpyutils import util

class Basic(unittest.TestCase):
  def setUp(self):
    self.test_dir = os.path.dirname(__file__)

  def testRandomSubsetUpToN(self):
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

if __name__ == '__main__':
  unittest.main()