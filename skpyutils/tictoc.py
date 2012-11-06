import time

class TicToc:
  """
  MATLAB-like tic/toc.
  """
  
  def __init__(self):
    self.labels = {}

  def tic(self,label=None):
    """
    Start timer for given label.

    Args:
      label (string): optional label for the timer.

    Returns:
      self
    """
    if not label:
      label = '__default'
    self.labels[label] = time.time()
    return self

  def toc(self,label=None,quiet=False):
    """
    Return elapsed time for given label.

    Args:
      label (string): [optional] label for the timer.

      quiet (boolean): [optional] print time elapsed if false

    Returns:
      elapsed (float): time elapsed
    """
    if not label:
      label = '__default'
    assert(label in self.labels)
    elapsed = time.time()-self.labels[label]
    name = " (%s)"%label if label else ""
    if not quiet:
      print "Time elapsed%s: %.3f"%(name,elapsed)
    return elapsed
  
  def qtoc(self,label=None):
    """
    Call toc(label, quiet=True).
    """
    return self.toc(label,quiet=True)

  def running(self, label=None, msg=None, interval=1):
    """
    Print <msg> every <interval> seconds, running the timer for <label>.

    Args:
      label (string): [optional] label for the timer

      msg (string): [optional] message to print

      interval (int): [optional] print every <interval> seconds

    Return
      self

    Raises
      none
    """
    if label not in self.labels:
      tt.tic(label)
    if tt.qtoc(label) > 1:
      print()
      tt.tic(label)
