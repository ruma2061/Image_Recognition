import cv2 as cv
import numpy as np

haystack_img = cv.imread(r"C:\Users\russe\Desktop\Image_Recognition\DeepCImages\haystackitems.png", cv.IMREAD_UNCHANGED)
needle_img = cv.imread(r"C:\Users\russe\Desktop\Image_Recognition\DeepCImages\fullbass.png", cv.IMREAD_UNCHANGED)

# List of 6 methods to use
matches = ['cv.TM_CCOEFF','cv.TM_CCOEFF_NORMED','cv.TM_CCORR','cv.TM_CCORR_NORMED','cv.TM_SQDIFF','cv.TM_SQDIFF_NORMED']

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

# Get the best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

print ('Best match top left position: %s' % str(max_loc))
print ('Best match confidence: %s' % max_val)

threshold = 0.8
if max_val >= threshold:
    print('Found needle in haystack')

    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    top_left = max_loc
    bottom_right = (top_left[0] + needle_img.shape[1], top_left[1] + needle_img.shape[0])

    # Draw rectangle on result image
    cv.rectangle(haystack_img, top_left, bottom_right, (0, 255, 0), thickness=2, lineType=cv.LINE_4)


    cv.imwrite(r"result.jpg", haystack_img)
    #cv.imshow('Result', haystack_img)
    #cv.waitKey()

else:
    print('Needle not found in haystack')

# Draw rectangle on result image