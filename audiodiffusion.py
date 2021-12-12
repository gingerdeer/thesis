import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq



# import the audio data
# split left and right channels
samplerate, data = wavfile.read("gradusample.wav")
actualsignal = data[:,0] #lc
actualsignal2 = data[:,1] #rc


# normalize the data

actualsignal = actualsignal / np.max(actualsignal)
actualsignal2 = actualsignal2 / np.max(actualsignal2)

# compute the convolution kernel

llambda = 1.0
t = np.linspace(-15, 15, 64)
kernel = np.exp(-1.0*abs(t)/(np.sqrt(llambda)))
kernel /= np.trapz(kernel) # normalize the integral to 1

# Uncomment to plot initial samples
#
#
#plt.clf()
#fig, ax = plt.subplots()
 #       plt.plot(xs,img_rst)
#plt.plot(actualsignal[:50])
#fig.savefig("actualsignal.png")
#plt.close("all")

# initialize

#img_rst5 = cv2.filter2D(img_src,-1,kernel5)
img_rst = actualsignal
img_rst2 = actualsignal2
M =0 
M2 = 0
delta_t = 0.1
direp = -1.0
picnum = 1

# pick a point to visualize the wave form
plotl = 0 #160000
ploth = plotl + 2048 #160600

maxsign = np.max(actualsignal)

# perform the iteration
# 
depth = 35
print( "Calculating diffusion scale set, " + "llambda = "+str(llambda)+
      ", iter_depth = "+ str(depth) +", iter count = "+ str(4 * depth)+ ", delta_t = "+ str(delta_t))
num_it = depth
for i in range(num_it):
    conv_step = np.convolve(img_rst,kernel,'same')
    conv_step2 = np.convolve(img_rst2,kernel,'same')
    M = max(M, np.max(conv_step))
    M2 = max(M2, np.max(conv_step2))
    delta_t = min(0.1, 2.5/M)
    delta_t2 = min(0.1, 2.5/M2)
    img_rst = img_rst+ (delta_t / llambda) * direp * conv_step    
    img_rst2 = img_rst2+ (delta_t2 / llambda) * direp * conv_step2    
#    img_rst = maxsign * ( img_rst / np.max(img_rst) )
    img_rst = img_rst / np.max(img_rst)
    img_rst2 = img_rst2 / np.max(img_rst2)
    if i%2 == 0: 
        plt.clf()
        fig, ax = plt.subplots()
 #       plt.plot(xs,img_rst)
        plt.plot(img_rst[plotl:ploth])
        #plt.axis('off')
        fig.savefig("fig"+str(picnum)+".png")
        picnum = picnum + 1
        plt.close("all")
        sys.stdout.write("."+str(picnum)+".")
# save the processed audio

wavfile.write("out.wav",samplerate,np.column_stack((img_rst,img_rst2)))    
plt.close("all")


# plot fourier spectrum 
plt.figure()
plt.title("Original signal")
data = actualsignal
samples=len(data)
datafft = fft(data)
#Get the absolute value of real and complex component:
fftabs = abs(datafft)
freqs = fftfreq(samples,1/samplerate)
plt.xlim( [10, samplerate/2] )
plt.xscale( 'log' )
plt.grid( True )
plt.xlabel( 'Frequency (Hz)' )
plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])

# plot fourier spectrum of result
plt.figure()
plt.title("Processed signal")
data = img_rst
samples=len(data)
datafft = fft(data)
#Get the absolute value of real and complex component:
fftabs = abs(datafft)
freqs = fftfreq(samples,1/samplerate)
plt.xlim( [10, samplerate/2] )
plt.xscale( 'log' )
plt.grid( True )
plt.xlabel( 'Frequency (Hz)' )
plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])
