import cv2
import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional

class BilliardsDetector:
    def __init__(self, monitor: Dict[str, int], masks: Dict[str, List[int]],
                 ball_params: Dict[str, Dict[str, Any]], pocket_params: Dict[str, Any],
                 line_params: Dict[str, Any]):
        
        self.monitor = monitor
        self.masks = masks
        self.ball_params = ball_params
        self.pocket_params = pocket_params
        self.line_params = line_params
        
        # GPU mats
        h, w = monitor["height"], monitor["width"]
        self.gpu_frame_bgra = cv2.cuda_GpuMat(h, w, cv2.CV_8UC4)  # type: ignore
        self.gpu_frame_bgr = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)  # type: ignore
        self.gpu_hsv = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)  # type: ignore
        self.gpu_mask = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)  # type: ignore
        
        # GPU detectors
        default_ball = ball_params.get('cue_ball', {})
        self.gpu_detector_balls = cv2.cuda.createHoughCirclesDetector(
            dp=1.0, minDist=default_ball.get('minDist', 20),
            cannyThreshold=default_ball.get('cannyThreshold', 60),
            votesThreshold=int(default_ball.get('votesThreshold', 8)),
            minRadius=default_ball.get('minRadius', 12),
            maxRadius=default_ball.get('maxRadius', 17)
        )
        
        self.gpu_detector_pockets = cv2.cuda.createHoughCirclesDetector(
            dp=1.0, minDist=pocket_params.get('minDist', 50),
            cannyThreshold=pocket_params.get('cannyThreshold', 100),
            votesThreshold=int(pocket_params.get('votesThreshold', 20)),
            minRadius=pocket_params.get('minRadius', 20),
            maxRadius=pocket_params.get('maxRadius', 40)
        )
        
        # Cached elements
        self.cached_felt_mask_gpu: Optional[cv2.cuda_GpuMat] = None  # type: ignore
        self.play_area_roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
        
        # Ghost line pixel count for display
        self.last_ghost_line_pixel_count = 0
        
        # Ball limits
        self.ball_limits: Dict[str, int] = {
            'cue_ball': 1, 'ghost_ball': 1, 'solids': 7, 'stripes': 7, 'eight_ball': 1
        }
    
    def upload_frame(self, frame: np.ndarray) -> bool:
        """Upload frame to GPU and convert to HSV"""
        try:
            self.gpu_frame_bgra.upload(frame)
            cv2.cuda.cvtColor(self.gpu_frame_bgra, cv2.COLOR_BGRA2BGR, self.gpu_frame_bgr)
            cv2.cuda.cvtColor(self.gpu_frame_bgr, cv2.COLOR_BGR2HSV, self.gpu_hsv)
            return True
        except cv2.error as e:
            print(f"âœ— GPU upload error: {e}")
            return False
    
    def detect_balls(self) -> Tuple[List[Dict[str, Any]], Optional[Tuple[int, int]]]:
        """Detect all balls using GPU Hough circles"""
        ball_types = ['cue_ball', 'ghost_ball', 'solids', 'stripes', 'eight_ball']
        detections = []
        ghost_ball_pos = None

        for name in ball_types:
            if name not in self.masks:
                continue

            params = self.ball_params[name]
            self.gpu_detector_balls.setMinDist(params['minDist'])
            self.gpu_detector_balls.setCannyThreshold(params['cannyThreshold'])
            self.gpu_detector_balls.setVotesThreshold(int(params['votesThreshold']))
            self.gpu_detector_balls.setMinRadius(params.get('minRadius', 1))
            self.gpu_detector_balls.setMaxRadius(params.get('maxRadius', 100))

            vals = self.masks[name]
            lower = tuple(float(x) for x in vals[:3])
            upper = tuple(float(x) for x in vals[3:6])

            # Use ROI if play area is cached, otherwise use full frame
            roi_offset_x = 0
            roi_offset_y = 0

            if self.play_area_roi is not None:
                # Extract ROI from GPU HSV image
                x, y, w, h = self.play_area_roi
                roi_offset_x = x
                roi_offset_y = y

                # Crop GPU mat to ROI
                gpu_hsv_roi = self.gpu_hsv.rowRange(y, y + h).colRange(x, x + w)
                gpu_mask_roi = self.gpu_mask.rowRange(y, y + h).colRange(x, x + w)

                cv2.cuda.inRange(gpu_hsv_roi, lower, upper, gpu_mask_roi)
                circles_gpu = self.gpu_detector_balls.detect(gpu_mask_roi)
            else:
                # Fallback to old behavior (full frame with optional mask)
                cv2.cuda.inRange(self.gpu_hsv, lower, upper, self.gpu_mask)

                if self.cached_felt_mask_gpu is not None and not self.cached_felt_mask_gpu.empty():
                    cv2.cuda.bitwise_and(self.gpu_mask, self.cached_felt_mask_gpu, self.gpu_mask)  # type: ignore

                circles_gpu = self.gpu_detector_balls.detect(self.gpu_mask)
            
            if not circles_gpu.empty():
                result = circles_gpu.download()
                if result.shape[0] > 0 and result.shape[1] > 0:
                    result = np.around(result).astype(int)

                    limit = self.ball_limits.get(name, 999)
                    count = 0

                    # Original iteration pattern
                    for circle in result[0]:
                        if count >= limit:
                            break

                        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

                        # Adjust coordinates back to full frame if using ROI
                        x += roi_offset_x
                        y += roi_offset_y

                        # Second-stage filtering by area
                        min_area = params.get('min_area', 0)
                        max_area = params.get('max_area', 5000)
                        area = np.pi * (r ** 2)

                        if not (min_area <= area <= max_area):
                            continue

                        detection_data = {
                            'name': name,
                            'x': x,
                            'y': y,
                            'r': r
                        }

                        detections.append(detection_data)
                        count += 1

                        if name == 'ghost_ball':
                            ghost_ball_pos = (detection_data['x'], detection_data['y'])
        
        return detections, ghost_ball_pos
    
    def detect_ghost_line_pixels(self, ghost_center: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """Detect white line pixels near ghost ball for trajectory fitting"""
        if 'ghost_line' not in self.masks or ghost_center is None:
            return None

        vals = self.masks["ghost_line"]
        lower = tuple(float(x) for x in vals[:3])
        upper = tuple(float(x) for x in vals[3:6])
        
        gx, gy = ghost_center
        region_size = self.line_params.get('ghost_line_roi', 65)
        min_line_gap = 5 # Minimum distance between ghostball and ghost line center

        # Define ROI around the ghost ball
        x1_roi = max(int(gx - region_size), 0)
        y1_roi = max(int(gy - region_size), 0)
        w_roi = min(int(gx + region_size), self.monitor['width']) - x1_roi
        h_roi = min(int(gy + region_size), self.monitor['height']) - y1_roi

        if w_roi <= 0 or h_roi <= 0:
            return None

        # Create mask only within the ROI for performance
        gpu_hsv_roi = self.gpu_hsv.rowRange(y1_roi, y1_roi + h_roi).colRange(x1_roi, x1_roi + w_roi)
        gpu_mask_roi = self.gpu_mask.rowRange(y1_roi, y1_roi + h_roi).colRange(x1_roi, x1_roi + w_roi)
        cv2.cuda.inRange(gpu_hsv_roi, lower, upper, gpu_mask_roi)
        
        mask_roi = gpu_mask_roi.download()

        # Find the center of the white pixels in the ROI
        white_points = cv2.findNonZero(mask_roi)
        if white_points is None or len(white_points) < 30: # Use a threshold for robustness
            self.last_ghost_line_pixel_count = 0
            return None

        white_points_array = white_points.reshape(-1, 2)
        center_x_roi = int(np.mean(white_points_array[:, 0]))
        center_y_roi = int(np.mean(white_points_array[:, 1]))

        # Direction is from ghost ball center to the center of the white line pixels
        direction = ( (center_x_roi + x1_roi) - gx, (center_y_roi + y1_roi) - gy )
        self.last_ghost_line_pixel_count = len(white_points)

        return direction
    
    def detect_pockets_from_mask(self, mask_cpu: np.ndarray) -> List[Dict[str, Any]]:
        """Detect pockets using contour analysis"""
        contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_pockets = []
        min_area = self.pocket_params.get('min_area', 500)
        min_circularity_scaled = self.pocket_params.get('min_circularity', 70) / 100.0
        max_area = self.pocket_params.get('max_area', 5000)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity > min_circularity_scaled:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        radius = int(math.sqrt(area / math.pi))
                        detected_pockets.append({
                            'contour': cnt,
                            'center': (cx, cy),
                            'radius': radius
                        })
        return detected_pockets
    
    def apply_rail_margins(self, mask: np.ndarray) -> np.ndarray:
        """Applies the configured edge margins to a mask to isolate rails."""
        modified_mask = mask.copy()
        margin_left = self.line_params.get('edge_margin_left', 0)
        margin_right = self.line_params.get('edge_margin_right', 0)
        margin_top_left = self.line_params.get('edge_margin_top_left', 0)
        margin_top_right = self.line_params.get('edge_margin_top_right', 0)
        margin_bottom_left = self.line_params.get('edge_margin_bottom_left', 0)
        margin_bottom_right = self.line_params.get('edge_margin_bottom_right', 0)

        height, width = modified_mask.shape
        if margin_left > 0:
            modified_mask[:, :margin_left] = 0
        if margin_right > 0:
            modified_mask[:, width-margin_right:] = 0
        if margin_top_left > 0:
            modified_mask[:margin_top_left, :width//2] = 0
        if margin_top_right > 0:
            modified_mask[:margin_top_right, width//2:] = 0
        if margin_bottom_left > 0:
            modified_mask[height-margin_bottom_left:, :width//2] = 0
        if margin_bottom_right > 0:
            modified_mask[height-margin_bottom_right:, width//2:] = 0
        
        return modified_mask
    
    def detect_rails_from_mask(self, mask_cpu: np.ndarray, frame_width: int, frame_height: int) -> List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, int]]]:
        """Detect rails using contour fitting with edge margins and line shortening"""
        
        contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rails = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(cnt)
            x_min_bbox, y_min_bbox = x_bbox, y_bbox
            x_max_bbox, y_max_bbox = x_bbox + w_bbox, y_bbox + h_bbox
            
            if abs(vx) > abs(vy):
                rails.append(((x_min_bbox, int(y.item())), (x_max_bbox, int(y.item()))))
            else:
                rails.append(((int(x.item()), y_min_bbox), (int(x.item()), y_max_bbox)))
        
        # Apply gap merging
        max_rail_gap = self.line_params.get('max_rail_gap', 10)
        if max_rail_gap > 0 and len(rails) > 1:
            rails = self._merge_rail_gaps(rails, max_rail_gap)
        
        # Apply line shortening
        shorten_pixels = self.line_params.get('rail_line_shorten_pixels', 0)
        if shorten_pixels > 0:
            rails = self._shorten_rails(rails, shorten_pixels)
        
        # Apply rail offset
        rail_offset = self.line_params.get('rail_offset_pixels', 0)
        if rail_offset != 0:
            rails = self._offset_rails(rails, rail_offset, frame_width, frame_height)
        
        # Add normals
        rails_with_normals = []
        for (p1, p2) in rails:
            normal = self._get_rail_normal(p1, p2, frame_width, frame_height)
            if normal is not None:
                rails_with_normals.append(((p1, p2), normal))
        
        return rails_with_normals
    
    def _merge_rail_gaps(self, rails: List, max_gap: int) -> List:
        """Merge broken rail segments with tolerance for position and angle"""
        if len(rails) <= 1:
            return rails
        
        merge_tolerance_px = self.line_params.get('merge_tolerance_px', 10)
        merge_tolerance_angle = self.line_params.get('merge_tolerance_angle', 5)
        
        merged_rails = []
        used_indices = set()
        
        for i in range(len(rails)):
            if i in used_indices:
                continue
            
            line1_p1, line1_p2 = rails[i]
            
            # Calculate angle of line1
            dx1 = line1_p2[0] - line1_p1[0]
            dy1 = line1_p2[1] - line1_p1[1]
            angle1 = math.degrees(math.atan2(dy1, dx1)) % 180
            
            extended_line_points = [line1_p1, line1_p2]
            
            for j in range(i + 1, len(rails)):
                if j in used_indices:
                    continue
                
                line2_p1, line2_p2 = rails[j]
                
                # Calculate angle of line2
                dx2 = line2_p2[0] - line2_p1[0]
                dy2 = line2_p2[1] - line2_p1[1]
                angle2 = math.degrees(math.atan2(dy2, dx2)) % 180
                
                # Check angle similarity
                angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
                if angle_diff > merge_tolerance_angle:
                    continue
                
                # Check distance between endpoints
                dist_p1p1 = np.linalg.norm(np.array(line1_p1) - np.array(line2_p1))
                dist_p1p2 = np.linalg.norm(np.array(line1_p1) - np.array(line2_p2))
                dist_p2p1 = np.linalg.norm(np.array(line1_p2) - np.array(line2_p1))
                dist_p2p2 = np.linalg.norm(np.array(line1_p2) - np.array(line2_p2))
                
                min_dist = min(dist_p1p1, dist_p1p2, dist_p2p1, dist_p2p2)
                
                if min_dist < max(max_gap, merge_tolerance_px):
                    extended_line_points.extend([line2_p1, line2_p2])
                    used_indices.add(j)
            
            # Compute merged line
            all_x = [p[0] for p in extended_line_points]
            all_y = [p[1] for p in extended_line_points]
            
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            if abs(max_x - min_x) > abs(max_y - min_y):
                merged_rails.append(((min_x, int(np.mean(all_y))), (max_x, int(np.mean(all_y)))))
            else:
                merged_rails.append(((int(np.mean(all_x)), min_y), (int(np.mean(all_x)), max_y)))
            
            used_indices.add(i)
        
        return merged_rails
    
    def _shorten_rails(self, rails: List, shorten_pixels: int) -> List:
        """Shorten rail lines by specified pixels from each end"""
        shortened_rails = []
        
        for (p1, p2) in rails:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length <= 2 * shorten_pixels:
                # Line too short to shorten - skip it
                continue
            
            # Unit vector
            ux = dx / length
            uy = dy / length
            
            # Shorten from both ends
            new_p1 = (int(p1[0] + ux * shorten_pixels), int(p1[1] + uy * shorten_pixels))
            new_p2 = (int(p2[0] - ux * shorten_pixels), int(p2[1] - uy * shorten_pixels))
            
            shortened_rails.append((new_p1, new_p2))
        
        return shortened_rails
    
    def _offset_rails(self, rails: List, offset_pixels: int, frame_width: int, frame_height: int) -> List:
        """Offset rails inward (positive) or outward (negative) from their detected position"""
        offset_rails = []
        
        for (p1, p2) in rails:
            is_horizontal = abs(p1[1] - p2[1]) < abs(p1[0] - p2[0])
            center_x = (p1[0] + p2[0]) / 2
            center_y = (p1[1] + p2[1]) / 2
            
            if is_horizontal:
                # Horizontal rail - shift vertically
                if center_y < frame_height / 2:
                    # Top rail - positive offset moves down (inward)
                    new_y = p1[1] + offset_pixels
                else:
                    # Bottom rail - positive offset moves up (inward)
                    new_y = p1[1] - offset_pixels
                offset_rails.append(((p1[0], new_y), (p2[0], new_y)))
            else:
                # Vertical rail - shift horizontally
                if center_x < frame_width / 2:
                    # Left rail - positive offset moves right (inward)
                    new_x = p1[0] + offset_pixels
                else:
                    # Right rail - positive offset moves left (inward)
                    new_x = p1[0] - offset_pixels
                offset_rails.append(((new_x, p1[1]), (new_x, p2[1])))
        
        return offset_rails
    
    def _get_rail_normal(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                         frame_width: int, frame_height: int) -> Tuple[int, int]:
        """Calculate inward-pointing normal for a rail"""
        is_horizontal = abs(p1[1] - p2[1]) < abs(p1[0] - p2[0])
        center_x = (p1[0] + p2[0]) / 2
        center_y = (p1[1] + p2[1]) / 2
        
        if is_horizontal:
            if center_y < frame_height / 2:
                return (0, 1)  # Top rail, points down
            else:
                return (0, -1)  # Bottom rail, points up
        else:
            if center_x < frame_width / 2:
                return (1, 0)  # Left rail, points right
            else:
                return (-1, 0)  # Right rail, points left
    
    def cache_felt_mask(self) -> bool:
        """Cache felt mask to GPU for ball detection masking and extract play area ROI"""
        if "felt" not in self.masks:
            print("âœ— Felt mask not configured")
            return False

        vals = self.masks["felt"]
        lower = tuple(float(x) for x in vals[:3])
        upper = tuple(float(x) for x in vals[3:6])

        cv2.cuda.inRange(self.gpu_hsv, lower, upper, self.gpu_mask)
        self.cached_felt_mask_gpu = cv2.cuda_GpuMat()  # type: ignore
        self.gpu_mask.copyTo(self.cached_felt_mask_gpu)

        # Apply morphological operations to get clean play area
        temp_mask = self.gpu_mask.download()
        kernel = np.ones((3,3), np.uint8)
        processed_mask = cv2.erode(temp_mask, kernel, iterations=1)
        processed_mask = cv2.dilate(processed_mask, kernel, iterations=1)

        # Find largest contour to get play area bounding box
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            self.play_area_roi = (x, y, w, h)
            print(f"âœ“ Cached felt mask (GPU+CPU) with play area ROI: {self.play_area_roi}")
        else:
            print("âœ“ Cached felt mask (GPU+CPU) - no contours found for ROI")

        return True
    
    def get_frame(self) -> np.ndarray:
        """Download current frame from GPU"""
        return self.gpu_frame_bgr.download()
