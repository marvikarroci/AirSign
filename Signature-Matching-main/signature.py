import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np


def match(img1, img2):

    bounded_sig1 = bounding(img1)
    bounded_sig2 = bounding(img2)

    # resize images for comparison
    img1 = rescale(bounded_sig1, 300, 300)
    img2 = rescale(bounded_sig2, 300, 300)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # display both images
    cv2.imshow("Signature 1", img1)
    cv2.imshow("Signature 2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #similarity_value = "{:.2f}".format(ssim(img1, img2) * 100)
    #return float(similarity_value)

    # Calculate the total number of black pixels that match in both images
    matching_black_pixels = np.sum((img1 == 0) & (img2 == 0))
    # Calculate the total number of black pixels in the first image
    total_black_pixels_img1 = np.sum(img1 == 0)

    # Calculate the similarity based on matching black pixels
    if total_black_pixels_img1 != 0:
        similarity_value = (matching_black_pixels / total_black_pixels_img1) * 100
        return round(float(similarity_value), 1)
    else:
        similarity_value = 0  # Avoid division by zero if there are no black pixels in img1
        return round(float(similarity_value), 1)


def rescale(image, width, height):
    # Calculate the aspect ratio of the original image
    aspect_ratio = image.shape[1] / image.shape[0]

    # Calculate the new width and height while maintaining the aspect ratio
    if width is None:
        new_width = int(height * aspect_ratio)
    elif height is None:
        new_height = int(width / aspect_ratio)
    else:
        new_width = width
        new_height = height

    # Resize the image to the new width and height
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


# schneidet das Bild mit der Unterschrift so zu, dass sie das gesamte Bild ausfuellt
def bounding(image):
    # alles was in diesen Farbraum [0,0,0] und [10,10,10] liegt wird gewertet als Teil der Signature
    lower = np.array([0, 0, 0])
    upper = np.array([10, 10, 10])
    mask = cv2.inRange(image, lower, upper)

    result = cv2.bitwise_and(image, image, mask=mask)
    result[mask == 0] = (255, 255, 255)

    # Find contours on extracted mask, combine boxes, and extract ROI
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts = np.concatenate(cnts)
    x, y, w, h = cv2.boundingRect(cnts)
    result = result[y:y + h, x:x + w]
    return result

