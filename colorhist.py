import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv
from math import pi

def colorhist(img):
  """
  Take an image as 3-d array of RGB channels and plot the distribution of hues
  on a color wheel.
  Return the figure with the plot.
  """
  if img.ndim < 3 or img.shape[2]<1:
    print("CAnnot process black and white images yet")
    return None

  img_hsv = rgb2hsv(img)
  # Only consider pixels with saturation > 0.2 and 0.15 < lightness < 0.95
  mask = (img_hsv[:,:,1]>0.2) & (img_hsv[:,:,2]>0.15) & (img_hsv[:,:,2]<0.95)
  filtered_hue = img_hsv[mask,0]

  h = np.round(filtered_hue*255).astype(int).flatten()
  hist,bins = np.histogram(h,256,[0,256])
  hist = 1.*hist / hist.max()
  polar_bins = 2*pi*bins/256

  cm = plt.get_cmap('hsv')
  hsv_colors = [cm(i) for i in range(256)]

  fig = plt.figure(figsize=(6,6),dpi=300)
  ax = fig.add_subplot(111,polar=True)
  # "colorbar"
  ax.vlines(2*pi-polar_bins[:-1],1,1.2,color=hsv_colors,linewidth=6)
  # actual values
  ax.vlines(2*pi-polar_bins[:-1],1-hist,np.ones(256),color=hsv_colors,linewidth=2)
  ax.set_ylim([0,1.2])
  ax.set_xticks([])
  ax.set_yticks([1])
  ax.set_yticklabels([])
  ax.grid(True,which="major",ls="-",lw=1,color="k")
  return fig

if __name__ == '__main__':
  img = imread('test/c.png')
  imshow(img);plt.show()
  fig = colorhist(img)
  plt.show(fig)
  plt.clf()

  img = imread('test/c2.png')
  imshow(img);plt.show()
  fig = colorhist(img)
  plt.show(fig)
  plt.clf()
  import sys; sys.exit()
