import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy.linalg
import sys
import scipy.linalg as sl
from scipy import integrate

#read image
#
img_src = cv2.imread('baboon.png')

# Uncomment to perform rescaling if necessary
#img_src = cv2.resize(img_src, (0, 0), fx=0.5, fy=0.5)


# Definitions for integration of the K kernel
epsilon=0.0001
N = 256
grid = np.linspace(epsilon,100,N)

def integralvectr(x,t,n=1,llambda=1.0):
    return np.exp(-t-((sl.norm(x)**2)/(4.0*t*llambda)))/(t**(n/2.0))

def integratedvec(x,n=1,llambda=1.0,grid=np.linspace(0.0001,100,256)):
    return (1.0/((4.0*llambda*np.pi)**(n/2.0))) * integrate.quad(lambda tt: integralvectr(x,tt,n=n,llambda=llambda), 0.0, np.inf)[0] # * np.trapz(integralvectr(x,grid,n=n,llambda=llambda))
    #return (1.0/((4.0*llambda*np.pi)**(n/2))) * np.trapz(integralvectr(x,grid,n=n,llambda=llambda))

def f(x,y):
    return integratedvec(np.array([x,y]),n=2,llambda=llambda)

def create_k_matrix(n,llambda=1.0):
    W = n
    convmat = np.zeros((2*W+1,2*W+1))
    #C = 0.99
    C = integrate.nquad(lambda a,b: integratedvec(np.array([a,b]),n=2.0,llambda=llambda), [[-W - 0.5,W + 0.5],[-W - 0.5,W + 0.5]])[0]
    for i in range(2*W+1):
        for j in range(2*W+1):
            x = i - W
            y = j - W
            convmat[i,j] = (1/C)  * integrate.nquad(lambda a,b : integratedvec(np.array([a,b]),n=2,llambda=llambda) , [[y-0.5,y+0.5],[x-0.5,x+0.5]])[0]
    return convmat

# Set parameters for the method and calculate the kernel
# Recommended values:
# W = 5
# llambda = 1.0
#

W = 5
#llambda = 10.0
llambda = 1.0
kernel5 = create_k_matrix(W,llambda=llambda)

# Perform the iteration
# 
# Parameters:
# num_it = number of iterations
# printmod = save an image every this many iterations
# direp = 1 for diffusion, -1 for sharpening
num_it = 4000
#print(kernel5)
M = 0
direp = 1.0
picnum= 1
img_rst5 = img_src
printmod = 50
for i in range(num_it):
    conv_step = cv2.filter2D(img_rst5,-1,kernel5) - img_rst5
    M = max(M, np.max(conv_step))
    delta_t = min(0.1, 2.5/M)
    img_rst5 = img_rst5 + (delta_t / llambda) * direp * conv_step  
    #
    #img_rst5 = np.max(img_src) * (img_rst5/np.max(img_rst5))
    if i%printmod == 0:      
        cv2.imwrite('lorebackforth' + str(picnum) +  '.png',img_rst5)
        picnum = picnum + 1
        sys.stdout.write(".")  

#plt.imshow(img_rst5)

