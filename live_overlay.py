import cv2
import numpy as np
import mss
import time

# Monitor configuration
MONITOR = {"top": 75, "left": 2151, "width": 1380, "height": 750}

def main():
    with mss.mss() as sct:
        cv2.namedWindow('Live Overlay', cv2.WINDOW_NORMAL)

        print("Press 'q' to quit.")

        # FPS Limiting
        target_fps = 45
        frame_time = 1.0 / target_fps

        while True:
            t_start = time.time()

            # 1. Capture Screen
            sct_img = sct.grab(MONITOR)
            image = np.array(sct_img)
            # mss returns BGRA, convert to BGR for processing
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            # 2. Identify Pink Ghostball
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Use calibration values from the original script or reasonable defaults
            lower_pink = np.array([140, 50, 50])
            upper_pink = np.array([179, 255, 255])

            mask_pink = cv2.inRange(hsv_image, lower_pink, upper_pink)

            kernel = np.ones((1,1), np.uint8)
            mask_pink_eroded = cv2.erode(mask_pink, kernel, iterations=1)
            mask_pink_cleaned = cv2.dilate(mask_pink_eroded, kernel, iterations=2)

            contours, _ = cv2.findContours(mask_pink_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour = None
            max_area = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_contour = contour

            ghostball_center = None

            # 3. Get Ghostball Center (CORRECTED with round)
            if largest_contour is not None:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    # Using round() instead of int() for better precision
                    cX = int(round(M["m10"] / M["m00"]))
                    cY = int(round(M["m01"] / M["m00"]))
                    ghostball_center = (cX, cY)

            # 4. Define Circular ROI
            image_roi = None
            roi_rect = None
            if ghostball_center is not None:
                center_x, center_y = ghostball_center
                roi_radius = 70

                x_bbox = max(0, int(center_x - roi_radius))
                y_bbox = max(0, int(center_y - roi_radius))
                x_bbox_end = min(image.shape[1], int(center_x + roi_radius))
                y_bbox_end = min(image.shape[0], int(center_y + roi_radius))

                w_bbox = x_bbox_end - x_bbox
                h_bbox = y_bbox_end - y_bbox

                roi_rect = (x_bbox, y_bbox, w_bbox, h_bbox)

                image_bbox = image[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]

                mask_circle = np.zeros(image_bbox.shape[:2], dtype=np.uint8)
                circle_center_bbox = (int(center_x - x_bbox), int(center_y - y_bbox))
                cv2.circle(mask_circle, circle_center_bbox, roi_radius, 255, -1)

                image_roi = cv2.bitwise_and(image_bbox, image_bbox, mask=mask_circle)

            # 5. Detect White Line (CORRECTED with fitLine)
            extended_line_endpoints = None

            if image_roi is not None:
                hsv_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
                lower_white = np.array([0, 0, 140])
                upper_white = np.array([180, 33, 255])

                mask_white_roi = cv2.inRange(hsv_roi, lower_white, upper_white)

                # Find all white pixels
                white_points = cv2.findNonZero(mask_white_roi)

                if white_points is not None and len(white_points) > 20: # Threshold to avoid noise
                    # Use fitLine for sub-pixel accuracy direction
                    vx, vy, x0, y0 = cv2.fitLine(white_points, cv2.DIST_L2, 0, 0.01, 0.01)

                    # Convert ROI coordinates to Global coordinates
                    x_roi, y_roi, _, _ = roi_rect

                    # The point (x0, y0) is relative to the ROI top-left
                    # We need a point on the line in global space
                    # Note: fitLine returns normalized vector (vx, vy)

                    global_x0 = x0[0] + x_roi
                    global_y0 = y0[0] + y_roi

                    # Ensure direction is correct (away from ghostball or through it?)
                    # The white line is usually an aiming guide extending FROM the cue ball (ghost ball position)
                    # We want to extend the line in both directions or just one?
                    # Let's project 'forward' and 'backward' from the ghostball center using this vector

                    # Better yet, enforce that the line passes through the ghostball center
                    # if we assume the ghostball IS the origin of the line.
                    # BUT, fitLine gives the best fit for the visible white pixels.
                    # Let's trust the fitLine direction, but anchor it to the ghostball center
                    # if the user wants it "originating" there, or just draw the fitted line infinite.

                    # Let's draw the fitted line extending across the screen
                    # Calculate two points far away using the direction (vx, vy)

                    slope = vy[0] / vx[0] if vx[0] != 0 else 1e9

                    # Determine endpoints at image boundaries
                    h, w = image.shape[:2]

                    # We can use the ghostball center as a reference point on the line
                    # if we assume the line passes through it.
                    # If the white line is slightly offset from center, fitLine will capture that true position.
                    # Let's use the fitted point (global_x0, global_y0) as the anchor.

                    scale = 2000
                    p1 = (int(global_x0 - vx[0] * scale), int(global_y0 - vy[0] * scale))
                    p2 = (int(global_x0 + vx[0] * scale), int(global_y0 + vy[0] * scale))

                    extended_line_endpoints = (p1, p2)

            # 6. Overlay
            final_image = image.copy()

            if extended_line_endpoints is not None:
                cv2.line(final_image, extended_line_endpoints[0], extended_line_endpoints[1], (255, 0, 0), 1)

            if ghostball_center is not None:
                 cv2.circle(final_image, ghostball_center, 1, (0, 255, 0), -1)

            # FPS Calculation
            elapsed = time.time() - t_start
            fps = 1 / elapsed if elapsed > 0 else 0
            cv2.putText(final_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Live Overlay', final_image)

            # Cap Frame Rate
            elapsed_total = time.time() - t_start
            sleep_time = frame_time - elapsed_total
            if sleep_time > 0:
                time.sleep(sleep_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
