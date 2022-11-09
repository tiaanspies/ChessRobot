import cv2 as cv
import matplotlib.pyplot as plt 

img1 = cv.imread('Photos/surgicalTheatre_1.png')
img2 = cv.imread('Photos/surgicalTheatre_2.png')
img3 = cv.imread('Photos/surgicalTheatre_3.png')


plt.figure("Surgical Theatre 1")
plt.title("Colour Intensities")
plt.xlabel("Colour Intensity")
plt.ylabel("Number of pixels")

########## IMAGE 1 ##########

cv.imshow("image1",img1)
hist_blue = cv.calcHist([img1],[0],None,[256],[0,256])
plt.plot(hist_blue, color = 'b')
hist_green = cv.calcHist([img1],[1],None,[256],[0,256])
plt.plot(hist_green, color = 'g')
hist_red = cv.calcHist([img1],[2],None,[256],[0,256])
plt.plot(hist_red, color = 'r')
plt.xlim([0,256])
#plt.show()


plt.figure("Surgical Theatre 2")
plt.title("Colour Intensities")
plt.xlabel("Colour Intensity")
plt.ylabel("Number of pixels")

########## IMAGE 2 ##########
cv.imshow("image2",img2)
hist_blue = cv.calcHist([img2],[0],None,[256],[0,256])
plt.plot(hist_blue, color = 'b')
hist_green = cv.calcHist([img2],[1],None,[256],[0,256])
plt.plot(hist_green, color = 'g')
hist_red = cv.calcHist([img2],[2],None,[256],[0,256])
plt.plot(hist_red, color = 'r')
plt.xlim([0,256])
#plt.show()

plt.figure("Surgical Theatre 3")
plt.title("Colour Intensities")
plt.xlabel("Colour Intensity")
plt.ylabel("Number of pixels")

########## IMAGE 3 ##########
cv.imshow("image3",img3)
hist_blue = cv.calcHist([img3],[0],None,[256],[0,256])
plt.plot(hist_blue, color = 'b')
hist_green = cv.calcHist([img3],[1],None,[256],[0,256])
plt.plot(hist_green, color = 'g')
hist_red = cv.calcHist([img3],[2],None,[256],[0,256])
plt.plot(hist_red, color = 'r')
plt.xlim([0,256])
plt.show()



cv.waitKey(0)