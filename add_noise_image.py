import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise, img_as_uint
from skimage.color import rgb2gray

N = 256
grid = np.zeros((N,N))
gridf = np.zeros((N,N))


img_src = cv2.imread('corrected_im_noise3.tif')
img_src = cv2.resize(img_src, (0, 0), fx=0.1, fy=0.1)


grid =  cv2.imread('baboon.png')
grid = img_src
img = rgb2gray(grid)

#plt.clf()
#fig, ax = plt.subplots()
#im = plt.imshow(grid)#,cmap = plt.get_cmap(colormap))

gauss = img_as_uint(random_noise(img, mode='gaussian',var=0.05))
poisson = img_as_uint(random_noise(img, mode='poisson'))
saltandpepper = img_as_uint(random_noise(img, mode='s&p',amount=0.1))
speckle = img_as_uint(random_noise(img, mode='speckle',var=0.05))

cv2.imwrite('noise_orig' +  '.png',img)
cv2.imwrite('noise_gauss' +  '.png',gauss)
cv2.imwrite('noise_poisson' +  '.png',poisson)
cv2.imwrite('noise_sp' +  '.png',saltandpepper)
cv2.imwrite('noise_speckle' +  '.png',speckle)
    

        
        
plt.imshow(img)
plt.imshow(gauss)
plt.imshow(poisson)
plt.imshow(saltandpepper)
plt.imshow(speckle)


