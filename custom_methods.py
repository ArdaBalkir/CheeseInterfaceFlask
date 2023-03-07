import cv2
import numpy as np

# Returns hole_number, hole_to_surface_ratio
def holiness(image_array):

    #img = cv2.imread(image_path)
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding to create a binary image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours in the binary image
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask image for holes
    mask = np.zeros_like(opening)

    # Iterate through the contours and draw them onto the mask
    for cnt in contours:
        cv2.drawContours(mask, [cnt], 0, 255, -1)

    # Invert the mask image to create a mask for the holes
    holes_mask = cv2.bitwise_not(mask)

    # BLOB DETECTOR OPTIMIZATION, this step is vital to produce an accurate-ish hole detector

    # Create SimpleBlobDetector parameters
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

    # Perform blob detection to identify the holes
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh)

    # Print the number of holes detected
    print("Number of holes detected:", len(keypoints))

    # Get the amount of white pixels on the mask image
    cheese_material_surface = np.count_nonzero(mask)

    # Combine the threshold image and mask image
    combined = cv2.bitwise_or(thresh, holes_mask)
    # This is the final image we can also return for demonstration
    
    # Calculate the hole-to-surface ratio
    holes_surface = np.count_nonzero(combined <= 100)
    hole_to_surface_ratio = holes_surface / cheese_material_surface

    # Print the hole-to-surface ratio
    print("Hole-to-surface ratio:", "{:.3f}".format(hole_to_surface_ratio))

    # Perform blob detection to identify the holes in the combined image
    # This step is a failsafe, 
    keypoints1 = detector.detect(combined) 

    # Print the number of holes detected in the combined image
    print("Number of holes detected in the combined image:", len(keypoints1))
    
    return len(keypoints), "{:.3f}".format(hole_to_surface_ratio)

