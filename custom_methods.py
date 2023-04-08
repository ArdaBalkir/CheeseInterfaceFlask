import cv2
import numpy as np

# Returns hole_number, hole_to_surface_ratio
def holiness(image_array):

    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding to create a binary image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up and create a mask and find the contours
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(opening)

    # Draw contour
    for cnt in contours:
        cv2.drawContours(mask, [cnt], 0, 255, -1)

    holes_mask = cv2.bitwise_not(mask)

    # BLOB DETECTOR OPTIMIZATION, this step is vital to produce an accurate-ish hole detector
    params = cv2.SimpleBlobDetector_Params()

    # Adjust parameters to detect more blobs
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 10
    params.minArea = 10
    params.minDistBetweenBlobs = 1
    params.filterByConvexity = False
    params.filterByCircularity = False
    params.filterByArea = False
    # Parameters are subject to change 
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh)
    
    #Debug log
    print("Number of holes detected:", len(keypoints))

    # Get the amount of white pixels on the mask image
    cheese_material_surface = np.count_nonzero(mask)

    # Combine the threshold image and mask image to get rid of the background
    combined = cv2.bitwise_or(thresh, holes_mask)
    # This is the final image we can also return for demonstration
    
    # Calculate the hole-to-surface ratio
    holes_surface = np.count_nonzero(combined <= 100)
    hole_to_surface_ratio = holes_surface / cheese_material_surface

    # Debug log
    print("Hole-to-surface ratio:", "{:.3f}".format(hole_to_surface_ratio))
    
    return len(keypoints), "{:.3f}".format(hole_to_surface_ratio)

# Returns sat and br disregarding the background
def get_saturation_and_brightness(image_array):

    img_hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    sat = img_hsv[:,:,1]
    bright = img_hsv[:,:,2]

    # Applying masks to Saturation and Brightness
    sat_mask = sat >= 10
    bright_mask = bright >= 30

    # Our limits have been chosen after consulting a colour picker
    # Masks are applied
    sat_filtered = sat[sat_mask]
    bright_filtered = bright[bright_mask]

    # The average/mean is calculated
    mean_sat = np.mean(sat_filtered)
    mean_bright = np.mean(bright_filtered)
    return round(mean_sat, 2), round(mean_bright, 2)

def impurities(image_array):

    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Canny edge detection with high and low thresholds
    edges = cv2.Canny(blurred, 10, 50, apertureSize=3)

    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area, in this case 20
    # which has been found through testing!
    big_contours = [c for c in contours if cv2.contourArea(c) > 20]

    # Print the number of detected big contours
    print(f"Number of detected big contours: {len(big_contours)}")

    return len(big_contours)
