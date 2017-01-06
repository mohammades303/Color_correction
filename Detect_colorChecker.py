import numpy as np
import imutils
import cv2
from scipy import ndimage
from joblib import Parallel, delayed

def paralell_search(search_scale, degree, gray, H_card, W_card, card):
    found = None
    if degree is not 0:
        gray_rot = ndimage.rotate(gray, degree)
    else:
        gray_rot = gray
    for i in range(len(search_scale)):

        scale2 = search_scale[i]
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray_rot, width = int(gray.shape[1] * scale2))
        r = gray_rot.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < H_card or resized.shape[1] < W_card:
            break
        # apply template matching to find the template in the image 
        result = cv2.matchTemplate(resized, card, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r, scale2)
    return(found)
    
    
# this function searches for color card ('card') in the image ('img').
# The sizes of 'card' and the card in the image are assumed to be almost similar and have similar orientation.  
# However, search_scale and search_degree can be passed to test ranges of different sizes and orientations
# suggested ranges: [0.9,1.1] for search_scale and [-2.5,2.5] degree for search_degree

def detect_card(img, card, search_scale, search_degree, num_cores):

    (H_card, W_card) = card.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect edges in the grayscale image
    edged = cv2.Canny(gray,40,50)
    # search for the best scale and rotation degree
    results = Parallel(n_jobs = num_cores, backend = "threading")(delayed(paralell_search)(search_scale, degree, edged, H_card, W_card, card) for degree in search_degree)
    maxVal_all, _, _, _  = zip(*results)
    # select the best scale and rotation degree based on the maximum correlation
    ind = np.argmax(maxVal_all)
    maxVal, maxLoc, r, SCALE  = results[ind]
    deg = search_degree[ind]
    # obtain Colorcard locations
    (startX, startY) = (int(round(maxLoc[0]*r)), int(round(maxLoc[1]*r)))
    (endX, endY) = (int(round((maxLoc[0] + W_card)*r)), int(round((maxLoc[1] + H_card) * r)))
    # rotate image if obtained card was rotated
    if deg is not 0:
        output_img = ndimage.rotate(img, deg)
    else:
        output_img = img
    # crop Colorcard
    output_img = output_img[startY:endY,startX:endX,:]

    return(output_img,maxVal,SCALE)





