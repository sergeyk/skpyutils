import operator
import types
import numpy as np

class Table:
  """
  An ndarray with associated column names and (optionally) row names.
  Methods for selecting by column and row names, as well as filtering by value
  and sorting by column.
  Array should be two-dimensional.
  """

  ###################
  # Init/Copy/Repr
  ###################
  def __init__(self,arr=None,cols=None,index=None,name=None):
    """
    If arr and cols are passed in, initialize with them by reference.
    If arr is None, initialize with np.array([]) which has shape (0,).
    - index gives names to the rows.
    - name is a place to keep some optional identifying information.
    """
    # Passed-in array can be None, or an empty (0,) array,
    # or an empty (0,N) array, or (M,), or (M,N).
    # The last two cases are good.
    # The (0,N) case is special and we maintain that shape.
    # Otherwise, we set to empty (0,) array.
    self.arr = np.array([])
    if arr != None:
      if arr.ndim == 2 and arr.shape[0]==0:
        self.arr = arr
      if arr.shape[0]>0:
        # convert (M,) arrays to (1,M) and leave (M,N) arrays alone
        self.arr = np.atleast_2d(arr)
    self.cols = cols
    self.index = index
    self.name = name

  @property
  def shape(self):
    "Return shape of the array."
    return self.arr.shape

  def ind(self,col_name):
    "Return index of the given column name."
    return self.cols.index(col_name)

  def __copy__(self):
    "Make a copy of the Table and return it."
    arr = self.arr.copy() if not self.arr == None else None
    cols = list(self.cols) if not self.cols == None else None
    index = list(self.index) if hasattr(self,'index') and not self.index == None else None
    return Table(arr,cols,index,self.name) 
  def copy(self):
    return self.__copy__()

  def __repr__(self):
    # TODO: print index?
    return """
Table name: %(name)s | size: %(shape)s
%(cols)s
%(arr)s
"""%dict(self.__dict__.items()+{'shape':self.shape}.items())

  def __eq__(self,other):
    "Two Tables are equal if all columns and their names are equal, in order."
    ret = np.all(self.arr==other.arr) and \
          (self.cols == other.cols)
    # TODO
    #if hasattr(self,'index') and hasattr(other,'index'):
      #ret = ret and (self.index == other.index)
    return ret

  def sum(self,dim=0):
    "Return sum of the array along given dimension."
    return np.sum(self.arr,dim)

  ###################
  # Save/Load
  ###################
  def save_csv(self,filename):
    "Write array to file in csv format."
    with open(filename,'w') as f:
      f.write("%s\n"%','.join(self.cols))
      if hasattr(self,'index'):
        f.write("%s\n"%','.join(self.index))
      f.write("%s\n"%self.name)
      np.savetxt(f, self.arr, delimiter=',')

  @classmethod
  def load_from_csv(cls,filename):
    """Creates a new Table object by reading in a csv file with header."""
    table = Table()
    with open(filename) as f:
      table.cols = f.readline().strip().split(',')
      table.name = f.readline().strip()
    table.arr = np.loadtxt(filename, delimiter=',', skiprows=2)
    assert(len(table.cols) == table.arr.shape[1])
    return table

  ###################
  # Selection/Filtering
  ###################
  def subset(self,names_or_inds_or_mask,axis=1):
    """
    Return copy of Table with only the specified
    - columns, if axis==1;
    - or rows, if axis==0

    Acceptable inputs:
    - a name or list of names;
    - an index or list or ndarray of indices;
    - a list or ndarray of Boolean values.

    In the former two cases, the copy will return Table with the
    columns or rows in given order.
    """
    arr,cols,index = self.subset_arr_and_cols_and_index(names_or_inds_or_mask,axis)
    return Table(arr,cols,index,self.name)

  def subset_arr(self, names_or_inds_or_mask, axis=1):
    """
    Like subset(), but only returns the corresponding array.
    If the array to be returned has 1 as one of its dimensions,
    returns an (N,) array instead.
    """
    if self.arr.size == 0:
      return self.arr
    arr = self.subset_arr_and_cols_and_index(names_or_inds_or_mask,axis)[0]
    # squeeze() has a weird behavior where if the array has noly one element,
    # it will return a ()-sized array. We always want to return (N,).
    if max(arr.shape)==1: # in other words, if arr.shape==(1,1) or (1,)
      return arr[0]
    else:
      return arr.squeeze()

  def subset_arr_and_cols_and_index(self, names_or_inds_or_mask, axis):
    "Helper method to subset() and subset_arr()."
    # If the argument is not a list or array, make it a list
    index = None
    if not isinstance(names_or_inds_or_mask, np.ndarray) and \
       not isinstance(names_or_inds_or_mask, types.ListType):
      names_or_inds_or_mask = [names_or_inds_or_mask]

    # bool is a subclass of int, so this check gets both ints and booleans
    if isinstance(names_or_inds_or_mask[0],types.IntType):
      inds = names_or_inds_or_mask
    # we also support floats, for backwards compatibility reasons
    elif isinstance(names_or_inds_or_mask[0],types.FloatType):
      inds = [int(x) for x in names_or_inds_or_mask]
    elif isinstance(names_or_inds_or_mask[0],types.StringType):
      if axis==0:
        assert(self.index)
        inds = [self.index.index(name) for name in names_or_inds_or_mask]
      else:
        inds = [self.cols.index(name) for name in names_or_inds_or_mask]
    else:
      raise RuntimeError('names_or_inds_or_mask must be a list of one of those three!')

    if axis==0:
      cols = self.cols
      if hasattr(self,'index'):
        index = np.array(self.index)[inds].tolist() if self.index else None
      arr = self.arr[inds,:]
    else:
      cols = np.array(self.cols)[inds].tolist()
      if hasattr(self,'index'):
        index = self.index
      arr = self.arr[:,inds]
    return (arr,cols,index)

  def row_subset(self,names_or_inds_or_mask):
    "Return Table with only the specified rows. See subset()."
    return self.subset(names_or_inds_or_mask,axis=0)

  def row_subset_arr(self,names_or_inds_or_mask):
    "Like row_subset() but only return the corresponding array."
    return self.subset_arr(names_or_inds_or_mask,axis=0)

  def sort_by_column(self,col_name,descending=False):
    "Return copy of self with array sorted by column."
    col = self.arr[:,self.cols.index(col_name)]
    col = -col if descending else col
    inds = col.argsort()
    self.arr = self.arr[inds]
    if hasattr(self,'index'):
      self.index = np.array(self.index)[inds].tolist() if self.index else None
    return self

  def filter_on_column(self, col_name, val=True, op=operator.eq, omit=False):
    """
    Take name of column and value to filter by, and return
    copy of self with only the rows that satisfy the filter.
    Default value is True.
    By providing an operator, more than just equality filtering can be done.
    If omit, removes that column from the returned copy.
    """
    if col_name not in self.cols:
      warnings.warn("Column name not found in the Table.")
      return self
    if self.shape[0] < 1:
      return self
    table = self.copy()
    col_ind = table.cols.index(col_name)
    mask = op(table.arr[:,col_ind], val)
    table.arr = table.arr[mask,:]
    table.index = np.array(table.index)[mask].tolist() if table.index else None
    if omit:
      return table.with_column_omitted(col_name)
    return table

  def with_column_omitted(self,col_name):
    "Return Table with given column omitted. Not necessarily a copy."
    drop_mask = np.arange(self.shape[1])==self.cols.index(col_name)
    if self.arr.size > 0:
      arr = self.arr[:,-drop_mask]
    else:
      arr = self.arr
    cols = list(self.cols)
    cols.remove(col_name)
    return Table(arr,cols,self.index,self.name)

  def append_column(self,col_name,vals):
    "Return Table that is self with added given column at the end."
    if isinstance(vals,list):
      vals = np.array(vals)
    assert(vals.ndim==1 and vals.shape[0]==self.shape[0])
    table = self.copy()
    table.cols = self.cols+[col_name]
    table.arr = np.hstack((self.arr,np.atleast_2d(vals).T))
    return table
