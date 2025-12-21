import cv2
import numpy as np
import mss
import time
import json

class GhostballTracker:
    def __init__(self):
        # Your coordinates
        self.monitor = {"left": 2151, "top": 80, "width": 1380, "height": 770}
        self.prev_time = 0
        self.search_radius = 80 # How many pixels around the ball to search
        self.min_line_gap = 5 # Minimum distance between ghostball and ghost line center
        self.last_white_pixel_count = 0 # Store last white pixel count for display
        self.last_ghostball_pixel_count = 0 # Store last ghostball pixel count for display

        # Load HSV values from config
        with open('billiards_calibration.json', 'r') as f:
            config = json.load(f)
        ghost_line_vals = config['masks'].get('ghost_line', [0, 0, 162, 180, 34, 255])
        self.h_min = ghost_line_vals[0]
        self.s_min = ghost_line_vals[1]
        self.v_min = ghost_line_vals[2]
        self.h_max = ghost_line_vals[3]
        self.s_max = ghost_line_vals[4]
        self.v_max = ghost_line_vals[5]

        # Load ghostball HSV values from config
        ghostball_vals = config['masks'].get('ghost_ball', [140, 192, 82, 170, 255, 255])
        self.gb_h_min = ghostball_vals[0]
        self.gb_s_min = ghostball_vals[1]
        self.gb_v_min = ghostball_vals[2]
        self.gb_h_max = ghostball_vals[3]
        self.gb_s_max = ghostball_vals[4]
        self.gb_v_max = ghostball_vals[5]

        # Load rails cache from config
        self.rails = config.get('cached_elements', {}).get('rails', [])
        print(f"Loaded {len(self.rails)} rails from config")

        # Pre-allocate numpy arrays for HSV bounds (performance optimization)
        self.lower_ghostball = np.array([self.gb_h_min, self.gb_s_min, self.gb_v_min], dtype=np.uint8)
        self.upper_ghostball = np.array([self.gb_h_max, self.gb_s_max, self.gb_v_max], dtype=np.uint8)
        self.lower_white = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        self.upper_white = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)

        self.show_overlays = True

        # Create main window as resizable
        cv2.namedWindow('Ghostball Tracker', cv2.WINDOW_NORMAL)

    def line_intersection(self, p1, p2, p3, p4):
        # Find intersection point between line (p1,p2) and line segment (p3,p4)
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= u <= 1 and t > 0:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        return None

    def reflect_vector(self, dx, dy, normal):
        # Reflect direction vector (dx, dy) across a normal vector
        nx, ny = normal
        dot = dx * nx + dy * ny
        rx = dx - 2 * dot * nx
        ry = dy - 2 * dot * ny
        return rx, ry

    def draw_one_way_ray(self, img, start_pt, through_pt, color=(0,0,0)):
        # Draws a 'Laser' starting at start_pt, going through through_pt with rail bounces.
        h, w = img.shape[:2]
        x1, y1 = start_pt
        x2, y2 = through_pt

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0: return

        # Normalize direction
        length = np.sqrt(dx*dx + dy*dy)
        dx /= length
        dy /= length

        # Start position
        current_x, current_y = x1, y1
        max_bounces = 2
        bounce_count = 0

        for _ in range(max_bounces + 1):
            # Create far endpoint for ray
            far_x = current_x + dx * 5000
            far_y = current_y + dy * 5000

            # Find closest rail intersection
            closest_dist = float('inf')
            closest_intersection = None
            closest_normal = None

            for rail in self.rails:
                rail_segment = rail[0]
                rail_normal = rail[1]
                p3 = tuple(rail_segment[0])
                p4 = tuple(rail_segment[1])

                intersection = self.line_intersection(
                    (current_x, current_y), (far_x, far_y), p3, p4
                )

                if intersection:
                    dist = np.sqrt((intersection[0] - current_x)**2 + (intersection[1] - current_y)**2)
                    if dist < closest_dist and dist > 1:  # Avoid same point
                        closest_dist = dist
                        closest_intersection = intersection
                        closest_normal = rail_normal

            # If no rail hit, ball went into a pocket (gap between rails)
            if closest_intersection is None:
                # Draw line to table edge and stop (ball fell in pocket)
                cv2.line(img, (int(current_x), int(current_y)), (int(far_x), int(far_y)), color, 1)
                break

            # Draw line segment to rail
            if bounce_count < max_bounces:
                cv2.line(img, (int(current_x), int(current_y)),
                        (int(closest_intersection[0]), int(closest_intersection[1])), color, 1)

                # Reflect direction
                dx, dy = self.reflect_vector(dx, dy, closest_normal)
                current_x, current_y = closest_intersection
                bounce_count += 1
            else:
                # Max bounces reached, draw final segment
                cv2.line(img, (int(current_x), int(current_y)),
                        (int(closest_intersection[0]), int(closest_intersection[1])), color, 1)
                break

    def process_frame(self, img):
        if img is None: return img

        display_img = img.copy()
        
        # Draw 1 pixel black line for each rail and 2 pixel thick dots on endpoints
        if self.show_overlays:
            for rail in self.rails:
                p1 = tuple(map(int, rail[0][0]))
                p2 = tuple(map(int, rail[0][1]))
                cv2.line(display_img, p1, p2, (0, 0, 0), 1)
                cv2.circle(display_img, p1, 2, (0, 0, 0), -1)
                cv2.circle(display_img, p2, 2, (0, 0, 0), -1)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_full, w_full = display_img.shape[:2]

        # --- 1. Detect Ghostball (The Anchor) FIRST ---
        # We need the ghostball location to know where to create the search window
        mask_ghostball = cv2.inRange(hsv_img, self.lower_ghostball, self.upper_ghostball)

        # Store ghostball pixel count for display
        self.last_ghostball_pixel_count = cv2.countNonZero(mask_ghostball)

        anchor_found = False
        anchor_center = None

        # Coordinates for Line Detection ROI (Region of Interest)
        roi_x, roi_y = 0, 0
        roi_w, roi_h = 0, 0
        roi_img_hsv = hsv_img # Default to full image if no ball found

        # Use moments on the entire mask to find the center (robust)
        if self.last_ghostball_pixel_count > 800:
            M = cv2.moments(mask_ghostball, binaryImage=True)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                anchor_center = (cX, cY)
                anchor_found = True

                # Draw ALL contours for ghostball to show what is being seen
                if self.show_overlays:
                    gb_contours, _ = cv2.findContours(mask_ghostball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if gb_contours:
                        cv2.drawContours(display_img, gb_contours, -1, (0, 255, 0), 1)

                # --- OPTIMIZATION: DEFINE SEARCH BOX ---
                # We only look for white lines inside a box around the ball
                roi_x = max(0, anchor_center[0] - self.search_radius)
                roi_y = max(0, anchor_center[1] - self.search_radius)
                roi_w = min(w_full, anchor_center[0] + self.search_radius) - roi_x
                roi_h = min(h_full, anchor_center[1] + self.search_radius) - roi_y

                # Crop HSV image to the search box
                roi_img_hsv = hsv_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # --- 2. White Line Detection (Inside ROI Only) ---
        # For detection: Apply mask on the CROPPED image (very fast)
        mask_white = cv2.inRange(roi_img_hsv, self.lower_white, self.upper_white)

        # Count pixels
        white_pixel_count = cv2.countNonZero(mask_white) if anchor_found else 0

        # Simple line detection: find white pixels and draw ray from ghostball through center
        if anchor_found and white_pixel_count > 30:
            # Use moments on the entire cropped mask to find the center (robust to fragmentation)
            M = cv2.moments(mask_white, binaryImage=True)

            if M["m00"] != 0:
                rel_cX = int(M["m10"] / M["m00"])
                rel_cY = int(M["m01"] / M["m00"])

                # Convert relative ROI coordinates to global
                center_x = rel_cX + roi_x
                center_y = rel_cY + roi_y

                # Draw ALL contours for white line
                if self.show_overlays:
                    white_contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if white_contours:
                        # Offset all contours
                        for contour in white_contours:
                            contour[:, :, 0] += roi_x
                            contour[:, :, 1] += roi_y
                        cv2.drawContours(display_img, white_contours, -1, (0, 255, 255), 1)

                # Calculate distance between ghostball and ghost line center
                distance = np.sqrt((center_x - anchor_center[0])**2 + (center_y - anchor_center[1])**2)

                # Only draw if the distance is greater than minimum gap
                if distance > self.min_line_gap and self.show_overlays:
                    # Draw Ray: Ghostball -> Through white line center -> Wall
                    self.draw_one_way_ray(display_img, anchor_center, (center_x, center_y))

        # Store white pixel count for display
        self.last_white_pixel_count = white_pixel_count

        return display_img

    def run(self):
        print(f"Capturing region: {self.monitor}")
        print("Press 'q' to quit. Press 'o' to toggle overlays.")

        # Limit to 45 FPS
        target_fps = 45
        frame_time = 1.0 / target_fps

        with mss.mss() as sct:
            while True:
                frame_start = time.time()

                sct_img = sct.grab(self.monitor)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Process frame
                processed_frame = self.process_frame(frame)

                fps = 1 / (time.time() - self.prev_time) if self.prev_time > 0 else 0
                h, w = processed_frame.shape[:2]

                # Display FPS and ghost line pixel count at bottom
                cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"GL Pixels: {self.last_white_pixel_count}", (200, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"GB Pixels: {self.last_ghostball_pixel_count}", (450, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Ghostball Tracker', processed_frame)

                # Cap at 45 FPS
                elapsed = time.time() - frame_start
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                self.prev_time = frame_start

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('o'):
                    self.show_overlays = not self.show_overlays
                    print(f"Overlays: {'ON' if self.show_overlays else 'OFF'}")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = GhostballTracker()
    tracker.run()
