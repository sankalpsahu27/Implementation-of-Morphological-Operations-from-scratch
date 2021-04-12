import cv2
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

im = np.zeros((10,10),dtype='uint8')
print(im);
plt.imshow(im)

im[0,1] = 1
im[-1,0]= 1
im[-2,-1]=1
im[2,2] = 1
im[5:8,5:8] = 1

print(im)
plt.imshow(im)

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
print(element)

ksize = element.shape[0]

height,width = im.shape[:2]

dilatedEllipseKernel = cv2.dilate(im, element)
print(dilatedEllipseKernel)
plt.imshow(dilatedEllipseKernel)

border = ksize//2
paddedIm = np.zeros((height + border*2, width + border*2))
paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 0)
paddedDilatedIm = paddedIm.copy()

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width=50
frame_height=50
out = cv2.VideoWriter('dilationScratch.avi',fourcc, 10, (frame_width,frame_height))


for h_i in range(border, height+border):
    for w_i in range(border,width+border):
        if im[h_i-border,w_i-border]:
            print("White Pixel Found @ {},{}".format(h_i,w_i))
            
        roi = paddedIm[h_i-border: h_i+border+1,  w_i-border: w_i+border+1]
        tmp = cv2.bitwise_and(roi,element)
        paddedDilatedIm[h_i,w_i] = np.amax(tmp)
        dilatedImage = paddedDilatedIm[border:border+height,border:border+width]
        dilatedImage = dilatedImage*255
        print(paddedIm)
        plt.imshow(dilatedImage);plt.show()

        
        # Resize output to 50x50 before writing it to the video
        resizedFrame = cv2.resize(dilatedImage, (50, 50))
        Image = cv2.cvtColor(resizedFrame,cv2.COLOR_GRAY2BGR)
        out.write(Image)

# Release the VideoWriter object
out.release()

# Display final image (cropped)
print(paddedIm)
plt.imshow(dilatedImage);plt.show()


ErodedEllipseKernel = cv2.erode(im, element)
print(ErodedEllipseKernel)
plt.imshow(ErodedEllipseKernel);



border = ksize//2
paddedIm = np.zeros((height + border*2, width + border*2))
paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 1)
paddedErodedIm = paddedIm.copy()
# Create a VideoWriter object

fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width=50
frame_height=50
out = cv2.VideoWriter('erosionScratch.avi',fourcc, 10, (frame_width,frame_height))

for h_i in range(border, height+border):
    for w_i in range(border,width+border):
        if im[h_i-border,w_i-border]:
            print("White Pixel Found @ {},{}".format(h_i,w_i))
            
        roi = paddedIm[h_i-border: h_i+border+1,  w_i-border: w_i+border+1]
        tmp = cv2.bitwise_not(roi,element)
        paddedErodedIm[h_i,w_i] = np.amax(tmp)
        ErodedImage = paddedErodedIm[border:border+height,border:border+width]
        ErodedImage = ErodedImage*255
        print(paddedIm)
        plt.imshow(ErodedImage);plt.show()
        # Resize output to 50x50 before writing it to the video
        resizedFrame = cv2.resize(ErodedImage, (50, 50),interpolation = cv2.INTER_CUBIC)
        Image = cv2.cvtColor(resizedFrame,cv2.COLOR_GRAY2BGR)
        out.write(Image)

# Release the VideoWriter object
out.release()


# Display final image (cropped)
print(paddedIm)
plt.imshow(ErodedImage);plt.show()







