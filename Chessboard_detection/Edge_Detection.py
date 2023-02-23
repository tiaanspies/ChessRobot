import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt


def fitKClusters(img, n_clusters):
    """
    Fit 4 k-means clusters to the image. Use HSV color scale
    Weighting can be used if pieces are on starting squares.
    To increase the weight of pieces.
    """
    # imgHSV = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    # imgHSV = cv.cvtColor(img, cv.COLOR_RGB2Lab)
    imgHSV = img

    # Get a block for each square and resize it to 32x32,
    # then stack back together for 256x256 image
    
    # reshape image into a single line for k means fitting
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1)
    imgReshaped = np.reshape(imgHSV, (imgHSV.shape[0]*imgHSV.shape[1], 3))

    kmeans.fit(imgReshaped)       

    return kmeans

def findClusterImg(kmeans, img):
    """
    Assigns pixels to their closest cluster.
    returns image with all pixels assigned to cluster
    """
    imgReshaped = np.reshape(img, (img.shape[0]*img.shape[1], 3))

    predictions = kmeans.predict(imgReshaped)
    clustersInt = kmeans.cluster_centers_.astype(np.uint8)

    newImg = [clustersInt[x] for x in predictions]
    newImg = np.reshape(newImg, (img.shape[0], img.shape[1], 3))
    
    # return cv.cvtColor(newImg, cv.COLOR_HSV2RGB)
    # return cv.cvtColor(newImg, cv.COLOR_Lab2RGB)
    return newImg

def cannyShapes(img):
    cannyEdges = cv.Canny(img,100,200)
    cannyEdges = cv.dilate(cannyEdges, (5,5))

    # ret, thresh = cv.threshold(img, 127, 255, 0)
    # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # i=0
    # newContours = []
    # max = 0
    # for contour in contours:
    
    #     # here we are ignoring first counter because 
    #     # findcontour function detects whole image as shape
    #     if i == 0:
    #         i = 1
    #         continue
    
    #     # cv2.approxPloyDP() function to approximate the shape
    #     approx = cv.approxPolyDP(
    #         contour, 0.05 * cv.arcLength(contour, True), True)
        
    #     area = cv.contourArea(contour)
    #     if area > max:
    #         max = area
    #     # putting shape name at center of each shape

    #     if len(approx) == 4 and area > 2000:
    #         newContours.append(contour)

        
    # print("Max area: ", max)

    # img = cv.drawContours(img, newContours, -1, (0,255,0), 2)
    # cv.imshow("1", thresh)

    cv.imshow("Canny edges dialted", cannyEdges)
    cv.waitKey()
    cv.destroyAllWindows()

def findContEdges(img):
    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    i=0
    newContours = []
    max = 0
    for contour in contours:
    
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue
    
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv.approxPolyDP(
            contour, 0.05 * cv.arcLength(contour, True), True)
        
        area = cv.contourArea(contour)
        if area > max:
            max = area
        # putting shape name at center of each shape

        if len(approx) == 4 and area > 2000:
            newContours.append(contour)

        
    print("Max area: ", max)

    img = cv.drawContours(color, newContours, -1, (0,255,0), 2)
    cv.imshow("1", thresh)
    cv.waitKey()
    cv.destroyAllWindows()

def otsu(img):
     # find normalized_histogram, and its cumulative distribution function
    hist = cv.calcHist([img],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    print( "{} {}".format(thresh,ret) )
    return otsu


def main():
    img = cv.imread('Chessboard_detection\TestImages\Temp\\full_top_moved_far.JPG')
    # img = cv.imread('Chessboard_detection\TestImages\Temp\\empty.JPG')
    img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    img = cv.resize(img, (640, 480))
    img = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
    # img = cv.medianBlur(img, 5)
    assert img is not None, "file could not be read, check with os.path.exists()"

    # kmeans = fitKClusters(img, n_clusters=6)
    # imgKmeans = findClusterImg(kmeans, img)
    # cv.imshow("clustered", imgKmeans)

    otsu0 = otsu(img[:,:,0])
    otsu1 = otsu(img[:,:,1])
    otsu2 = otsu(img[:,:,2])

    images = [otsu0, otsu1, otsu2,
                img[:, :, 0], img[:, :, 1], img[:, :, 2]]

    for i in range(2):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        # plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.imshow(images[i*3+2],'gray')
        # plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        # plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()