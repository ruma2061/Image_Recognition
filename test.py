import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


haystack_img = cv.imread(r"C:\Users\russe\Desktop\Image_Recognition\DeepCImages\haystackitems.png", cv.IMREAD_UNCHANGED)
needle_img = cv.imread(r"C:\Users\russe\Desktop\Image_Recognition\DeepCImages\fullbass.png", cv.IMREAD_UNCHANGED)

obj = 'bass'

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)
threshold = 0.4
locate = np.where( result >= threshold) # all locations with a confidence above threshold
locate = list(zip(*locate[::-1])) # reverse the x and y coordinates two arrays to one with xy tuples
print(locate)

if locate:
    print('Found needle in haystack')

    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]
    line_color = (0, 255, 0)
    line_type = cv.LINE_4

    for loc in locate:
        top_left = loc
        bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)
        haystack_img = cv.rectangle(haystack_img, top_left, bottom_right, line_color, line_type)
        cv.putText(haystack_img, obj, top_left, cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)

    cv.imwrite(r"result.jpg", haystack_img)
else:
    print('Needle not found in haystack')

# Get the best match position
#min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

#print ('Best match top left position: %s' % str(max_loc))
#print ('Best match confidence: %s' % max_val)