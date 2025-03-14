import numpy as np 
import cv2 as cv 

# Define function to calculate curvature
def calculate_curvature(x, y):
    # Fit a quadratic curve (2nd order polynomial)
    coeffs = np.polyfit(x, y, 2)
    
    # Curvature formula: Curvature = 2 * A / (1 + (2Ay + B)^2)^(3/2)
    A = coeffs[0]
    B = coeffs[1]
    curvature = abs(2 * A) / (1 + (2 * A * np.mean(y) + B)**2)**(3/2)
    
    return curvature

def sorting(centers_initial, centers_present):
    if len(centers_initial) == 0 or len(centers_present) == 0:
        print("No sorting performed.")
        return np.array([])

    centers_intermediate = np.ones((5, 2))
    for i in range(len(centers_initial)):
        for j in range(len(centers_present)):
            # calculating distance and judge
            if np.sqrt(np.sum(np.square(centers_initial[i]-centers_present[j]))) < 40:
                centers_intermediate[i] = centers_present[j]
                break
    centers_intermediate = centers_intermediate.astype(np.int16)
    return centers_intermediate


if __name__ == '__main__':
    file_path = 'airpods1.jpeg'
    img_rgb = cv.imread(file_path)

    if img_rgb is None:
        print("Error opening image file")
    else:
        # Blurring
        img_rgb = cv.medianBlur(img_rgb, 5)

        img_hsv = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV)

        # Detecting points by color
        red_hsv_lower = np.array([0, 50, 50])
        red_hsv_higher = np.array([10, 255, 255])
        mask1 = cv.inRange(img_hsv, lowerb=red_hsv_lower, upperb=red_hsv_higher)

        red_hsv_lower = np.array([156, 50, 50])
        red_hsv_higher = np.array([180, 255, 255])
        mask2 = cv.inRange(img_hsv, lowerb=red_hsv_lower, upperb=red_hsv_higher)
        mask = mask1 + mask2

        # Detecting contours
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Finding centers of contours
        centers = []
        for cnt in contours:
            (x, y), radius = cv.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            cv.circle(img_rgb, center, 5, (255, 0, 0), -1)
            centers.append([center[0], center[1]])

        centers = np.array(centers)
        # Sort centers by y-coordinate
        sorted_centers = centers[centers[:, 1].argsort()]

        # Calculate curvature
        x = sorted_centers[:, 0]
        y = sorted_centers[:, 1]
        curvature = calculate_curvature(x, y)
        print("Curvature:", curvature)

        for i in range(len(sorted_centers) - 1):
            cv.line(img_rgb, tuple(sorted_centers[i]), tuple(sorted_centers[i + 1]), (255, 0, 0), 2)
            

        cv.imshow('image', img_rgb)
        cv.waitKey(0)
        cv.destroyAllWindows()

