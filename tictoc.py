##############################################
# Tic/Toc
##############################################
class TicToc:
  "MATLAB tic/toc."
  def __init__(self):
    self.labels = {}

  def tic(self,label=None):
    "Start timer for given label. Returns self."
    if not label:
      label = '__default'
    self.labels[label] = time.time()
    return self

  def toc(self,label=None,quiet=False):
    "Return elapsed time for given label. Optionally print time."
    if not label:
      label = '__default'
    assert(label in self.labels)
    elapsed = time.time()-self.labels[label]
    if not quiet:
      print "Time elapsed: %.3f"%elapsed
    return elapsed
  
  def qtoc(self,label=None):
    "Quiet toc()"
    return self.toc(label,quiet=True)