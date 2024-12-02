import cv2 as cv
import numpy as np

# Load image as Numpy array in BGR order

file = r'C:\Users\russe\Desktop\Image_Recognition\DeepCImages\fullbass.png'
needle_img = cv.imread(file, cv.IMREAD_UNCHANGED)

# Extract alpha channel
if needle_img.shape[2] == 4:
    alphachannel = needle_img[:, :, 3]
    needle_img = needle_img[:, :, :3]
else:
    alphachannel = np.ones(needle_img.shape[:2], dtype=needle_img.dtype) * 255

# Load background image
haystack_img = cv.imread(r'C:\Users\russe\Desktop\Image_Recognition\DeepCImages\haystackitems.png', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

topleft = max_loc
bottomright = (topleft[0] + needle_img.shape[1], topleft[1] + needle_img.shape[0])
# Draw rectangle on result image
cv.rectangle(haystack_img, topleft, bottomright, (0, 255, 0), thickness=2, lineType=cv.LINE_4)


# Save result
cv.imwrite('result.png', result)