import cv2
import numpy as np

# Constants
OUTPUT_IMAGE_WIDTH = 500


def preprocess_and_detect_contours(input_image):
    """Preprocess the image and detect contours."""
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    #cv2.imshow("blurred_image", blurred_image)
    edges = cv2.Canny(blurred_image, 75, 200)
    #cv2.imshow("edges", edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area in descending order and take the largest ones
    return sorted(contours, key=cv2.contourArea, reverse=True)[:5]


def find_document_contour(contours):
    """Find the contour that most closely resembles a rectangle."""
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # 4 points imply a rectangle/polygon
            return approx
    return None


def order_points(points):
    """Reorder points to a consistent order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")  # Create an array to store the ordered points
    s = points.sum(axis=1)  # Calculate the sum of x and y values for each point
    diff = np.diff(points, axis=1)  # Calculate the difference between x and y for each point
    rect[0] = points[np.argmin(s)]  # Top-left point (smallest sum)
    rect[2] = points[np.argmax(s)]  # Bottom-right point (largest sum)
    rect[1] = points[np.argmin(diff)]  # Top-right point (smallest difference)
    rect[3] = points[np.argmax(diff)]  # Bottom-left point (largest difference)
    return rect  # Return the ordered points


def perspective_transform(input_image, contour):
    """Apply a perspective transformation to the document."""

    # The `contour` is reshaped into a 4x2 array representing the four corners of the detected document
    ordered_rect = order_points(contour.reshape(4, 2))
    (tl, tr, br, bl) = ordered_rect

    # Calculate width and height for the transformed image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Destination points for the warped image
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Perform the perspective transform
    transform_matrix = cv2.getPerspectiveTransform(ordered_rect, dst)
    return cv2.warpPerspective(input_image, transform_matrix, (max_width, max_height))


def resize_image(image, width):
    """Resize the given image maintaining aspect ratio."""
    aspect_ratio = width / image.shape[1]
    height = int(image.shape[0] * aspect_ratio)
    return cv2.resize(image, (width, height))


# Main processing steps
input_path = 'scanned-form.jpg'
input_image = cv2.imread(input_path)
if input_image is None:
    raise FileNotFoundError(f"Could not load image from {input_path}")

# Make a copy of the original image for transformations
original_image = input_image.copy()

# Detect contours
contours = preprocess_and_detect_contours(input_image)
#print(contours)

# Find the document contour
document_contour = find_document_contour(contours)

#print(document_contour)

if document_contour is not None:
    # Apply perspective transformation
    warped_image = perspective_transform(original_image, document_contour)

    # Resize the output image to a fixed width for consistency
    final_image = resize_image(warped_image, OUTPUT_IMAGE_WIDTH)

    # Save and display the result
    output_path = 'rectified_document.jpg'
    cv2.imwrite(output_path, final_image)
    print(f"Rectified document saved to {output_path}")
    cv2.imshow("Rectified Document", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Document contour could not be detected.")
