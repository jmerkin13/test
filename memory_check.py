import cv2
import numpy as np
import mss
import time
import os
from pynput import keyboard

# --- Configuration ---
MONITOR = {"top": 75, "left": 2151, "width": 1380, "height": 750}
WINDOW_NAME = "Pool Aim Detection"

# --- HSV Color Ranges ---
LOWER_PINK = np.array([140, 50, 50])
UPPER_PINK = np.array([179, 255, 255])
LOWER_WHITE = np.array([0, 0, 140])
UPPER_WHITE = np.array([180, 33, 255])

# --- Detection Parameters ---
ROI_RADIUS = 70
ENDPOINT_PROXIMITY_THRESHOLD = 5.0
HOUGH_THRESHOLD = 20
MIN_LINE_LENGTH = 35
MAX_LINE_GAP = 5

# --- Pre-calculated Kernel ---
# Pre-calculating this array prevents creating a new numpy array
# every single frame, reducing memory churn.
MORPH_KERNEL = np.ones((3, 3), np.uint8)

# --- Rail Endpoints (Inner 6 rails) ---
RAIL_ENDPOINTS = {
    "top_left_rail_start_x": 128,
    "top_left_rail_start_y": 81,
    "top_left_rail_end_x": 646,
    "top_left_rail_end_y": 81,
    "top_right_rail_start_x": 730,
    "top_right_rail_start_y": 81,
    "top_right_rail_end_x": 1246,
    "top_right_rail_end_y": 81,
    "bottom_left_rail_start_x": 126,
    "bottom_left_rail_start_y": 676,
    "bottom_left_rail_end_x": 647,
    "bottom_left_rail_end_y": 676,
    "bottom_right_rail_start_x": 734,
    "bottom_right_rail_start_y": 676,
    "bottom_right_rail_end_x": 1252,
    "bottom_right_rail_end_y": 676,
    "left_rail_start_x": 85,
    "left_rail_start_y": 121,
    "left_rail_end_x": 85,
    "left_rail_end_y": 632,
    "right_rail_start_x": 1293,
    "right_rail_start_y": 123,
    "right_rail_end_x": 1293,
    "right_rail_end_y": 628
}

# --- Pre-calculate Rails List ---
# Pre-calculating this list prevents creating a new list of dictionaries
# every single frame (45 times a second), reducing memory churn.
RAILS_LIST = [
    {
        "start": (RAIL_ENDPOINTS["top_left_rail_start_x"], RAIL_ENDPOINTS["top_left_rail_start_y"]),
        "end": (RAIL_ENDPOINTS["top_left_rail_end_x"], RAIL_ENDPOINTS["top_left_rail_end_y"]),
    },
    {
        "start": (RAIL_ENDPOINTS["top_right_rail_start_x"], RAIL_ENDPOINTS["top_right_rail_start_y"]),
        "end": (RAIL_ENDPOINTS["top_right_rail_end_x"], RAIL_ENDPOINTS["top_right_rail_end_y"]),
    },
    {
        "start": (RAIL_ENDPOINTS["bottom_left_rail_start_x"], RAIL_ENDPOINTS["bottom_left_rail_start_y"]),
        "end": (RAIL_ENDPOINTS["bottom_left_rail_end_x"], RAIL_ENDPOINTS["bottom_left_rail_end_y"]),
    },
    {
        "start": (RAIL_ENDPOINTS["bottom_right_rail_start_x"], RAIL_ENDPOINTS["bottom_right_rail_start_y"]),
        "end": (RAIL_ENDPOINTS["bottom_right_rail_end_x"], RAIL_ENDPOINTS["bottom_right_rail_end_y"]),
    },
    {
        "start": (RAIL_ENDPOINTS["left_rail_start_x"], RAIL_ENDPOINTS["left_rail_start_y"]),
        "end": (RAIL_ENDPOINTS["left_rail_end_x"], RAIL_ENDPOINTS["left_rail_end_y"]),
    },
    {
        "start": (RAIL_ENDPOINTS["right_rail_start_x"], RAIL_ENDPOINTS["right_rail_start_y"]),
        "end": (RAIL_ENDPOINTS["right_rail_end_x"], RAIL_ENDPOINTS["right_rail_end_y"]),
    },
]


def detect_ghostball(hsv_image):
    """Detect the pink ghostball and return its center coordinates."""
    mask_pink = cv2.inRange(hsv_image, LOWER_PINK, UPPER_PINK)

    # Use pre-calculated kernel
    mask_pink = cv2.erode(mask_pink, MORPH_KERNEL, iterations=1)
    mask_pink = cv2.dilate(mask_pink, MORPH_KERNEL, iterations=2)

    contours, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None

    cX = round(M["m10"] / M["m00"])
    cY = round(M["m01"] / M["m00"])

    return (cX, cY)


def get_circular_roi(image, center, radius):
    """Extract a circular ROI around the given center."""
    center_x, center_y = center
    img_h, img_w = image.shape[:2]

    x_bbox = max(0, center_x - radius)
    y_bbox = max(0, center_y - radius)
    x_bbox_end = min(img_w, center_x + radius)
    y_bbox_end = min(img_h, center_y + radius)

    w_bbox = x_bbox_end - x_bbox
    h_bbox = y_bbox_end - y_bbox

    # Check for invalid bounding box
    if w_bbox <= 0 or h_bbox <= 0:
        return None, None

    image_bbox = image[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]

    # Check if extracted image is valid
    if image_bbox.size == 0:
        return None, None

    mask_circle = np.zeros(image_bbox.shape[:2], dtype=np.uint8)
    circle_center_bbox = (round(center_x - x_bbox), round(center_y - y_bbox))
    cv2.circle(mask_circle, circle_center_bbox, radius, (255,), -1)

    image_roi = cv2.bitwise_and(image_bbox, image_bbox, mask=mask_circle)

    return image_roi, (x_bbox, y_bbox, w_bbox, h_bbox)


def detect_white_line(image_roi, ghostball_center, roi_rect):
    """Detect the white aim line in the ROI."""
    hsv_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv_roi, LOWER_WHITE, UPPER_WHITE)

    edges = cv2.Canny(mask_white, 1, 130, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                            minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)

    if lines is None:
        return None

    x_roi, y_roi, _, _ = roi_rect
    ghostball_center_roi = (float(ghostball_center[0] - x_roi),
                            float(ghostball_center[1] - y_roi))

    candidate_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1_f, y1_f, x2_f, y2_f = float(x1), float(y1), float(x2), float(y2)

        dist_to_ep1 = np.sqrt((ghostball_center_roi[0] - x1_f)**2 +
                              (ghostball_center_roi[1] - y1_f)**2)
        dist_to_ep2 = np.sqrt((ghostball_center_roi[0] - x2_f)**2 +
                              (ghostball_center_roi[1] - y2_f)**2)

        if dist_to_ep1 <= ENDPOINT_PROXIMITY_THRESHOLD or dist_to_ep2 <= ENDPOINT_PROXIMITY_THRESHOLD:
            seg_len_sq = (x2_f - x1_f)**2 + (y2_f - y1_f)**2

            if seg_len_sq == 0:
                min_dist = dist_to_ep1
            else:
                t = ((ghostball_center_roi[0] - x1_f) * (x2_f - x1_f) +
                     (ghostball_center_roi[1] - y1_f) * (y2_f - y1_f)) / seg_len_sq
                t = max(0.0, min(1.0, t))
                closest_x = x1_f + t * (x2_f - x1_f)
                closest_y = y1_f + t * (y2_f - y1_f)
                min_dist = np.sqrt((ghostball_center_roi[0] - closest_x)**2 +
                                   (ghostball_center_roi[1] - closest_y)**2)

            candidate_lines.append((min_dist, line[0]))

    if not candidate_lines:
        return None

    candidate_lines.sort(key=lambda x: x[0])
    best_line = candidate_lines[0][1]

    # Convert to global coordinates
    x1_global = best_line[0] + x_roi
    y1_global = best_line[1] + y_roi
    x2_global = best_line[2] + x_roi
    y2_global = best_line[3] + y_roi

    return (x1_global, y1_global, x2_global, y2_global)


def extend_line_to_boundary(line_coords, ghostball_center, img_shape):
    """Extend the detected line to the image boundary."""
    x1_seg, y1_seg, x2_seg, y2_seg = line_coords
    img_height, img_width = img_shape[:2]

    x1_f, y1_f = float(x1_seg), float(y1_seg)
    x2_f, y2_f = float(x2_seg), float(y2_seg)
    gb_x, gb_y = float(ghostball_center[0]), float(ghostball_center[1])

    # Determine which endpoint is closer to ghostball
    dist1 = np.sqrt((gb_x - x1_f)**2 + (gb_y - y1_f)**2)
    dist2 = np.sqrt((gb_x - x2_f)**2 + (gb_y - y2_f)**2)

    if dist1 < dist2:
        start_x, start_y = x1_f, y1_f
        other_x, other_y = x2_f, y2_f
    else:
        start_x, start_y = x2_f, y2_f
        other_x, other_y = x1_f, y1_f

    dx, dy = other_x - start_x, other_y - start_y
    magnitude = np.sqrt(dx**2 + dy**2)

    if magnitude == 0:
        return None

    dx /= magnitude
    dy /= magnitude

    min_t = float('inf')

    if abs(dx) < 1e-6:  # Vertical line
        if dy > 0:
            t = (float(img_height - 1) - start_y) / dy
        else:
            t = (0.0 - start_y) / dy
        if t > 0:
            min_t = t
    elif abs(dy) < 1e-6:  # Horizontal line
        if dx > 0:
            t = (float(img_width - 1) - start_x) / dx
        else:
            t = (0.0 - start_x) / dx
        if t > 0:
            min_t = t
    else:  # General case
        # Left boundary (x = 0)
        t_x0 = (0.0 - start_x) / dx
        if t_x0 > 0:
            y_at_x0 = start_y + t_x0 * dy
            if 0 <= y_at_x0 <= img_height - 1 and t_x0 < min_t:
                min_t = t_x0

        # Right boundary (x = img_width - 1)
        t_x_max = (float(img_width - 1) - start_x) / dx
        if t_x_max > 0:
            y_at_x_max = start_y + t_x_max * dy
            if 0 <= y_at_x_max <= img_height - 1 and t_x_max < min_t:
                min_t = t_x_max

        # Top boundary (y = 0)
        t_y0 = (0.0 - start_y) / dy
        if t_y0 > 0:
            x_at_y0 = start_x + t_y0 * dx
            if 0 <= x_at_y0 <= img_width - 1 and t_y0 < min_t:
                min_t = t_y0

        # Bottom boundary (y = img_height - 1)
        t_y_max = (float(img_height - 1) - start_y) / dy
        if t_y_max > 0:
            x_at_y_max = start_x + t_y_max * dx
            if 0 <= x_at_y_max <= img_width - 1 and t_y_max < min_t:
                min_t = t_y_max

    if min_t == float('inf'):
        return None

    extended_x = start_x + min_t * dx
    extended_y = start_y + min_t * dy

    return (round(start_x), round(start_y), round(extended_x), round(extended_y))


def draw_overlay(image, ghostball_center, white_line, extended_line, show_rails=False):
    """Draw the detection overlay on the image."""
    output = image.copy()

    # Draw rails if enabled (using pre-calculated global list)
    if show_rails:
        for rail in RAILS_LIST:
            start = rail["start"]
            end = rail["end"]
            # Draw rail line (1px thick, cyan)
            cv2.line(output, start, end, (255, 255, 0), 1)
            # Draw endpoints (small circles, 2px radius, cyan)
            cv2.circle(output, start, 2, (255, 255, 0), -1)
            cv2.circle(output, end, 2, (255, 255, 0), -1)

    # Draw extended line in blue
    if extended_line is not None:
        cv2.line(output, (extended_line[0], extended_line[1]),
                 (extended_line[2], extended_line[3]), (255, 0, 0), 1)

    # Draw detected white line segment in red
    if white_line is not None:
        cv2.line(output, (white_line[0], white_line[1]),
                 (white_line[2], white_line[3]), (0, 0, 255), 1)

    # Draw ghostball center in green
    if ghostball_center is not None:
        cv2.circle(output, ghostball_center, 2, (0, 255, 0), -1)

    return output


def main():
    # Create screenshots directory
    screenshot_dir = "screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    # Global keyboard listener state
    running = [True]  # Use list to allow modification in nested function
    screenshot_requested = [False]
    show_rails = [False]  # Toggle for rail overlay

    def on_key_press(key):
        """Handle global keyboard events"""
        try:
            if hasattr(key, 'char') and key.char in ['s', 'S']:
                screenshot_requested[0] = True
            elif hasattr(key, 'char') and key.char in ['q', 'Q']:
                running[0] = False
            elif hasattr(key, 'char') and key.char in ['r', 'R']:
                show_rails[0] = not show_rails[0]
                print(f"Rails overlay: {'ON' if show_rails[0] else 'OFF'}")
        except AttributeError:
            pass

    # Start global keyboard listener
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    print("Starting Pool Aim Detection...")
    print(f"Capturing from: {MONITOR}")
    print("Controls (GLOBAL - works even when window is not focused):")
    print("  'S' - Save screenshot")
    print("  'R' - Toggle rails overlay")
    print("  'Q' - Quit")
    print(f"Screenshots will be saved to: {screenshot_dir}/")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, MONITOR["width"], MONITOR["height"])

    fps_time = time.time()
    frame_count = 0
    fps = 0

    # Frame rate limiter
    TARGET_FPS = 45
    frame_time = 1.0 / TARGET_FPS

    try:
        with mss.mss() as sct:
            while running[0]:
                loop_start = time.time()
                # Capture screen
                screenshot = sct.grab(MONITOR)
                frame = np.array(screenshot)

                # Convert BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Convert to HSV for detection
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Detect ghostball
                ghostball_center = detect_ghostball(hsv_frame)

                white_line = None
                extended_line = None

                if ghostball_center is not None:
                    # Get circular ROI around ghostball
                    image_roi, roi_rect = get_circular_roi(frame, ghostball_center, ROI_RADIUS)

                    # Only proceed if ROI is valid
                    if image_roi is not None and roi_rect is not None:
                        # Detect white aim line
                        white_line = detect_white_line(image_roi, ghostball_center, roi_rect)

                        if white_line is not None:
                            # Extend line to boundary
                            extended_line = extend_line_to_boundary(white_line, ghostball_center, frame.shape)

                # Draw overlay
                output_frame = draw_overlay(frame, ghostball_center, white_line, extended_line, show_rails[0])

                # Check if screenshot was requested
                if screenshot_requested[0]:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{screenshot_dir}/screenshot_{timestamp}.png"
                    cv2.imwrite(filename, output_frame)
                    print(f"Screenshot saved: {filename}")
                    screenshot_requested[0] = False

                # Calculate and display FPS
                frame_count += 1
                if time.time() - fps_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    fps_time = time.time()

                cv2.putText(output_frame, f"FPS: {fps}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show frame
                cv2.imshow(WINDOW_NAME, output_frame)

                # Keep window responsive
                cv2.waitKey(1)

                # Enforce frame rate limit
                elapsed = time.time() - loop_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup guarantees
        print("Cleaning up resources...")
        try:
            listener.stop()
        except:
            pass
        cv2.destroyAllWindows()
        print("Pool Aim Detection stopped.")


if __name__ == "__main__":
    main()
