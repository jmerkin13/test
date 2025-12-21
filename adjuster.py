# adjuster.py - Calibration tool for billiards detection system
# Provides trackbar UI for tuning HSV masks and detector parameters
# Supports saving individual masks and caching static elements

import cv2
import mss
import numpy as np
import time
from typing import Dict, Any, Optional, List

from config import BilliardsConfig, get_default_ball_params, get_default_pocket_params, get_default_line_params
from detector import BilliardsDetector
from logic import BilliardsPhysics

class BilliardsAdjuster:
    def __init__(self, config_file: str = 'billiards_calibration.json'):
        # Load config
        self.config = BilliardsConfig(config_file)
        self.config.load()
        
        # Screen capture
        self.sct = mss.mss()
        self.monitor = {"left": 2151, "top": 80, "width": 1380, "height": 770}
        
        # Check CUDA
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            raise RuntimeError("No CUDA-enabled GPU found")
        print(f"✓ CUDA GPU: {cv2.cuda.getDevice()}")
        
        # Setup parameters
        self.ball_params = {}
        ball_types = ['cue_ball', 'ghost_ball', 'solids', 'stripes', 'eight_ball']
        loaded_ball_params = self.config.detector_params.get('balls', {})
        
        if 'minDist' in loaded_ball_params:
            for ball_type in ball_types:
                # Start with a copy of the shared params
                params = loaded_ball_params.copy()
                # Ensure area parameters from the root level are also included
                params.setdefault('min_area', loaded_ball_params.get('min_area', 500))
                params.setdefault('max_area', loaded_ball_params.get('max_area', 5000))
                self.ball_params[ball_type] = params
        else:
            # This path assumes a nested structure, which seems to be the new default
            for ball_type in ball_types:
                self.ball_params[ball_type] = loaded_ball_params.get(ball_type, get_default_ball_params())
        
        self.pocket_params = self.config.detector_params.get('pockets', get_default_pocket_params())
        self.line_params = self.config.detector_params.get('line', get_default_line_params())
        
        # Initialize detector
        self.detector = BilliardsDetector(
            self.monitor,
            self.config.masks,
            self.ball_params,
            self.pocket_params,
            self.line_params
        )
        
        # Load cached elements
        cached_rails = self.config.get_cached_rails()
        cached_pockets = self.config.get_cached_pockets()
        
        # Initialize physics
        self.physics = BilliardsPhysics(
            self.monitor,
            cached_rails,
            cached_pockets,
            max_bounces=2
        )
        
        # Tuning state
        self.current_mask_name: Optional[str] = None
        self.tuning_mode_active = False
        self.show_overlays = True
        self.bounces_enabled = True
        self.show_canny_preview = False # Toggle for ball canny preview
        self.live_tuning_pockets: Optional[List] = None
        self.last_smoothed_ball: Optional[Dict] = None
        self.ball_detection_count = 0
        self.line_detection_count = 0

        # Mask selection map
        self.mask_map: Dict[int, Optional[str]] = {
            ord('1'): 'cue_ball', ord('2'): 'ghost_ball', ord('3'): 'solids',
            ord('4'): 'stripes', ord('5'): 'eight_ball', ord('6'): 'pockets',
            ord('7'): 'rails', ord('8'): 'felt', ord('9'): 'ghost_line', ord('0'): None
        }
        
        # Colors
        self.colors = {
            'cue_ball': (255, 255, 255), 'ghost_ball': (0, 0, 0),
            'solids': (0, 0, 0), 'stripes': (0, 0, 0),
            'eight_ball': (0, 0, 0), 'pockets': (0, 0, 0),
            'rails': (0, 0, 0), 'felt': (0, 0, 0),
            'ghost_line': (0, 0, 0), 'cue_line': (0, 0, 0)
        }
        
        # Create resizable windows
        self.window_video = "Adjuster - Video Feed" 
        self.window_detector_controls = "B & P Adjuster - Detector Controls (Ball/Pocket)" 
        self.window_line_controls = "L Key Adjuster - Line Controls" 
        self.window_hsv_trackbars = " S key HSV Controls" 
        self.window_hsv_mask = "Mask Preview" 
        
        cv2.namedWindow(self.window_video, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_detector_controls, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_line_controls, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_hsv_trackbars, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_hsv_mask, cv2.WINDOW_NORMAL)

        cv2.resizeWindow(self.window_video, self.monitor["width"] , self.monitor["height"])
        cv2.resizeWindow(self.window_detector_controls, 350, 600) 
        cv2.resizeWindow(self.window_line_controls, 350, 350)
        cv2.resizeWindow(self.window_hsv_trackbars, 350, 250) 
        cv2.resizeWindow(self.window_hsv_mask, self.monitor["width"] // 2, self.monitor["height"] // 2)
        cv2.moveWindow(self.window_video, 0, 5)
        cv2.moveWindow(self.window_hsv_mask, 935, 744)
        cv2.moveWindow(self.window_detector_controls, 1264, 744)
        cv2.moveWindow(self.window_line_controls, 1593, 8)
        
        # Create trackbars
        self._create_trackbars()
        
        # Initial state setup
        self.current_mask_name = 'cue_ball'
        self.tuning_mode_active = True
        self._update_trackbar_positions()

        # Performance
        self.target_fps = 45
        self.frame_time = 1.0 / self.target_fps
    
    def _create_trackbars(self):
        # HSV trackbars
        cv2.createTrackbar('H_min', self.window_hsv_trackbars, 0, 179, lambda x: None)
        cv2.createTrackbar('S_min', self.window_hsv_trackbars, 0, 255, lambda x: None)
        cv2.createTrackbar('V_min', self.window_hsv_trackbars, 0, 255, lambda x: None)
        cv2.createTrackbar('H_max', self.window_hsv_trackbars, 179, 179, lambda x: None)
        cv2.createTrackbar('S_max', self.window_hsv_trackbars, 255, 255, lambda x: None)
        cv2.createTrackbar('V_max', self.window_hsv_trackbars, 255, 255, lambda x: None)
        
        # Ball detector trackbars
        default_ball = get_default_ball_params()
        cv2.createTrackbar('Ball_minDist', self.window_detector_controls, default_ball['minDist'], 200, lambda x: None)
        cv2.createTrackbar('Ball_Canny', self.window_detector_controls, default_ball['cannyThreshold'], 200, lambda x: None)
        cv2.createTrackbar('Ball_Votes', self.window_detector_controls, int(default_ball['votesThreshold'] * 2), 200, lambda x: None)
        cv2.createTrackbar('Ball_Outline_Thick', self.window_detector_controls, default_ball['lineThickness'], 20, lambda x: None)
        cv2.createTrackbar('Ball_minArea', self.window_detector_controls, 500, 5000, lambda x: None)
        cv2.createTrackbar('Ball_maxArea', self.window_detector_controls, 5000, 10000, lambda x: None)
        cv2.createTrackbar('Ball_minRadius', self.window_detector_controls, default_ball.get('minRadius', 12), 100, lambda x: None)
        cv2.createTrackbar('Ball_maxRadius', self.window_detector_controls, default_ball.get('maxRadius', 17), 100, lambda x: None)
        
        # Pocket detector trackbars
        cv2.createTrackbar('Pocket_minDist', self.window_detector_controls, self.pocket_params['minDist'], 200, lambda x: None)
        cv2.createTrackbar('Pocket_Canny', self.window_detector_controls, self.pocket_params['cannyThreshold'], 200, lambda x: None)
        cv2.createTrackbar('Pocket_Votes', self.window_detector_controls, int(self.pocket_params['votesThreshold'] * 2), 200, lambda x: None)
        cv2.createTrackbar('Pocket_minR', self.window_detector_controls, self.pocket_params['minRadius'], 100, lambda x: None)
        cv2.createTrackbar('Pocket_maxR', self.window_detector_controls, self.pocket_params['maxRadius'], 100, lambda x: None)
        cv2.createTrackbar('minArea', self.window_detector_controls, self.pocket_params.get('min_area', 500), 5000, lambda x: None)
        cv2.createTrackbar('minCircularity', self.window_detector_controls, self.pocket_params.get('min_circularity', 70), 100, lambda x: None)
        cv2.createTrackbar('maxArea', self.window_detector_controls, self.pocket_params.get('max_area', 5000), 10000, lambda x: None)

        # Line detector trackbars
        cv2.createTrackbar('angleTolerance', self.window_line_controls, self.line_params.get('angle_tolerance', 10), 45, lambda x: None)
        cv2.createTrackbar('Line_Thick', self.window_line_controls, self.line_params.get('line_thickness', 1), 20, lambda x: None)
        cv2.createTrackbar('ghostLineROI', self.window_line_controls, self.line_params.get('ghost_line_roi', 65), 200, lambda x: None)
        cv2.createTrackbar('edge_margin_left', self.window_line_controls, self.line_params.get('edge_margin_left', 0), 200, lambda x: None)
        cv2.createTrackbar('edge_margin_right', self.window_line_controls, self.line_params.get('edge_margin_right', 0), 200, lambda x: None)
        cv2.createTrackbar('edge_margin_top_left', self.window_line_controls, self.line_params.get('edge_margin_top_left', 0), 200, lambda x: None)
        cv2.createTrackbar('edge_margin_top_right', self.window_line_controls, self.line_params.get('edge_margin_top_right', 0), 200, lambda x: None)
        cv2.createTrackbar('edge_margin_bottom_left', self.window_line_controls, self.line_params.get('edge_margin_bottom_left', 0), 200, lambda x: None)
        cv2.createTrackbar('edge_margin_bottom_right', self.window_line_controls, self.line_params.get('edge_margin_bottom_right', 0), 200, lambda x: None)
    
    def _update_trackbar_positions(self):
        if not self.current_mask_name:
            return
        
        # HSV Trackbars
        hsv = self.config.masks.get(self.current_mask_name, [0,0,0,179,255,255])
        cv2.setTrackbarPos('H_min', self.window_hsv_trackbars, hsv[0])
        cv2.setTrackbarPos('S_min', self.window_hsv_trackbars, hsv[1])
        cv2.setTrackbarPos('V_min', self.window_hsv_trackbars, hsv[2])
        cv2.setTrackbarPos('H_max', self.window_hsv_trackbars, hsv[3])
        cv2.setTrackbarPos('S_max', self.window_hsv_trackbars, hsv[4])
        cv2.setTrackbarPos('V_max', self.window_hsv_trackbars, hsv[5])
        
        # Ball Detector Trackbars
        ball_types = ['cue_ball', 'ghost_ball', 'solids', 'stripes', 'eight_ball']
        if self.current_mask_name in ball_types:
            params = self.ball_params[self.current_mask_name]
            cv2.setTrackbarPos('Ball_minDist', self.window_detector_controls, params['minDist'])
            cv2.setTrackbarPos('Ball_Canny', self.window_detector_controls, params['cannyThreshold'])
            cv2.setTrackbarPos('Ball_Votes', self.window_detector_controls, int(params['votesThreshold'] * 2))
            cv2.setTrackbarPos('Ball_Outline_Thick', self.window_detector_controls, params.get('lineThickness', 2))
            cv2.setTrackbarPos('Ball_minArea', self.window_detector_controls, params.get('min_area', 500)) # Corrected key
            cv2.setTrackbarPos('Ball_maxArea', self.window_detector_controls, params.get('max_area', 5000)) # Corrected key
            cv2.setTrackbarPos('Ball_minRadius', self.window_detector_controls, params.get('minRadius', 12))
            cv2.setTrackbarPos('Ball_maxRadius', self.window_detector_controls, params.get('maxRadius', 17))

        # Pocket Detector Trackbars
        if self.current_mask_name == 'pockets':
            params = self.pocket_params
            cv2.setTrackbarPos('Pocket_minDist', self.window_detector_controls, params['minDist'])
            cv2.setTrackbarPos('Pocket_Canny', self.window_detector_controls, params['cannyThreshold'])
            cv2.setTrackbarPos('Pocket_Votes', self.window_detector_controls, int(params['votesThreshold'] * 2))
            cv2.setTrackbarPos('Pocket_minR', self.window_detector_controls, params['minRadius'])
            cv2.setTrackbarPos('Pocket_maxR', self.window_detector_controls, params['maxRadius'])
            cv2.setTrackbarPos('minArea', self.window_detector_controls, params.get('min_area', 500))
            cv2.setTrackbarPos('minCircularity', self.window_detector_controls, params.get('min_circularity', 70))
            cv2.setTrackbarPos('maxArea', self.window_detector_controls, params.get('max_area', 5000))

        # Line Detector Trackbars
        if self.current_mask_name in ['rails', 'ghost_line']:
            params = self.line_params
            cv2.setTrackbarPos('angleTolerance', self.window_line_controls, params.get('angle_tolerance', 10))
            cv2.setTrackbarPos('Line_Thick', self.window_line_controls, params.get('line_thickness', 2))
            cv2.setTrackbarPos('ghostLineROI', self.window_line_controls, params.get('ghost_line_roi', 65))
            cv2.setTrackbarPos('edge_margin_left', self.window_line_controls, params.get('edge_margin_left', 0))
            cv2.setTrackbarPos('edge_margin_right', self.window_line_controls, params.get('edge_margin_right', 0))
            cv2.setTrackbarPos('edge_margin_top_left', self.window_line_controls, params.get('edge_margin_top_left', 0))
            cv2.setTrackbarPos('edge_margin_top_right', self.window_line_controls, params.get('edge_margin_top_right', 0))
            cv2.setTrackbarPos('edge_margin_bottom_left', self.window_line_controls, params.get('edge_margin_bottom_left', 0))
            cv2.setTrackbarPos('edge_margin_bottom_right', self.window_line_controls, params.get('edge_margin_bottom_right', 0))
    
    def _get_trackbar_values(self):
        # HSV
        if self.current_mask_name:
            h_min = cv2.getTrackbarPos('H_min', self.window_hsv_trackbars)
            s_min = cv2.getTrackbarPos('S_min', self.window_hsv_trackbars)
            v_min = cv2.getTrackbarPos('V_min', self.window_hsv_trackbars)
            h_max = cv2.getTrackbarPos('H_max', self.window_hsv_trackbars)
            s_max = cv2.getTrackbarPos('S_max', self.window_hsv_trackbars)
            v_max = cv2.getTrackbarPos('V_max', self.window_hsv_trackbars)
            self.config.masks[self.current_mask_name] = [h_min, s_min, v_min, h_max, s_max, v_max]
            self.detector.masks = self.config.masks
        
        # Ball detector
        minDist = max(1, cv2.getTrackbarPos('Ball_minDist', self.window_detector_controls))
        canny = max(1, cv2.getTrackbarPos('Ball_Canny', self.window_detector_controls))
        votes = max(1, cv2.getTrackbarPos('Ball_Votes', self.window_detector_controls)) / 2.0
        min_area = max(1, cv2.getTrackbarPos('Ball_minArea', self.window_detector_controls))
        max_area = max(1, cv2.getTrackbarPos('Ball_maxArea', self.window_detector_controls))
        lineThick = max(1, cv2.getTrackbarPos('Ball_Outline_Thick', self.window_detector_controls))
        minRadius = max(1, cv2.getTrackbarPos('Ball_minRadius', self.window_detector_controls))
        maxRadius = max(1, cv2.getTrackbarPos('Ball_maxRadius', self.window_detector_controls))

        # Always update ball params from their trackbars, regardless of current mask
        ball_types = ['cue_ball', 'ghost_ball', 'solids', 'stripes', 'eight_ball']
        for ball_type in ball_types:
            import math
            params = self.ball_params[ball_type]
            params['minDist'] = minDist
            params['cannyThreshold'] = canny
            params['votesThreshold'] = votes
            params['min_area'] = min_area
            params['max_area'] = max_area
            params['lineThickness'] = lineThick
            params['minRadius'] = minRadius
            params['maxRadius'] = maxRadius

            # Ensure maxRadius is always greater than minRadius to prevent OpenCV assertion failure
            if params['maxRadius'] <= params['minRadius']: #
                params['maxRadius'] = params['minRadius'] + 1 #
        
        # Ensure the detector instance is updated with the new ball params
        self.detector.ball_params = self.ball_params
        
        # Pocket detector
        minDist_p = max(1, cv2.getTrackbarPos('Pocket_minDist', self.window_detector_controls))
        canny_p = max(1, cv2.getTrackbarPos('Pocket_Canny', self.window_detector_controls))
        votes_p = max(1, cv2.getTrackbarPos('Pocket_Votes', self.window_detector_controls)) / 2.0
        minR_p = max(1, cv2.getTrackbarPos('Pocket_minR', self.window_detector_controls))
        maxR_p = max(1, cv2.getTrackbarPos('Pocket_maxR', self.window_detector_controls))
        new_min_area = max(1, cv2.getTrackbarPos('minArea', self.window_detector_controls))
        new_max_area = max(1, cv2.getTrackbarPos('maxArea', self.window_detector_controls))
        new_circularity = cv2.getTrackbarPos('minCircularity', self.window_detector_controls)

        self.pocket_params['minDist'] = minDist_p
        self.pocket_params['cannyThreshold'] = canny_p
        self.pocket_params['votesThreshold'] = votes_p
        self.pocket_params['minRadius'] = minR_p
        self.pocket_params['maxRadius'] = maxR_p
        self.pocket_params['min_area'] = new_min_area
        self.pocket_params['max_area'] = new_max_area
        self.pocket_params['min_circularity'] = new_circularity
        
        self.detector.pocket_params = self.pocket_params
        
        # Line detector
        self.line_params['angle_tolerance'] = max(1, cv2.getTrackbarPos('angleTolerance', self.window_line_controls))
        self.line_params['line_thickness'] = max(1, cv2.getTrackbarPos('Line_Thick', self.window_line_controls))
        self.line_params['ghost_line_roi'] = max(10, cv2.getTrackbarPos('ghostLineROI', self.window_line_controls))
        self.line_params['edge_margin_left'] = cv2.getTrackbarPos('edge_margin_left', self.window_line_controls)
        self.line_params['edge_margin_right'] = cv2.getTrackbarPos('edge_margin_right', self.window_line_controls)
        self.line_params['edge_margin_top_left'] = cv2.getTrackbarPos('edge_margin_top_left', self.window_line_controls)
        self.line_params['edge_margin_top_right'] = cv2.getTrackbarPos('edge_margin_top_right', self.window_line_controls)
        self.line_params['edge_margin_bottom_left'] = cv2.getTrackbarPos('edge_margin_bottom_left', self.window_line_controls)
        self.line_params['edge_margin_bottom_right'] = cv2.getTrackbarPos('edge_margin_bottom_right', self.window_line_controls)
        self.detector.line_params = self.line_params

    def visualize(self, frame, detections, ghost_line, cue_line, ghost_ball_pos) -> np.ndarray:
        #Draw overlays relevant to current tuning mask 
        if not self.show_overlays:
            return frame
        
        # Determine what to show based on current mask
        show_rails = True
        show_pockets = True
        show_ghost_roi = False
        show_cue_line = True
        show_ghost_line = True
        
        if self.tuning_mode_active and self.current_mask_name:
            # Context-aware overlay filtering
            if self.current_mask_name == 'felt':
                # Only show felt area (no other overlays)
                show_rails = False
                show_pockets = False
                show_cue_line = False
                show_ghost_line = False
            
            elif self.current_mask_name == 'rails':
                # Show only rails being detected
                show_pockets = False
                show_cue_line = False
                show_ghost_line = False
                show_rails = True

            
            elif self.current_mask_name == 'pockets':
                # Show pockets and rails for context
                show_rails = True
                show_pockets = True
                show_cue_line = False
                show_ghost_line = False
            
            elif self.current_mask_name == 'ghost_line':
                # Show ghost line ROI, cue line, and trajectory
                show_ghost_roi = True
                show_rails = False
                show_pockets = False
                show_cue_line = True
                show_ghost_line = True
            
            elif self.current_mask_name in ['cue_ball', 'ghost_ball', 'solids', 'stripes', 'eight_ball']:
                # Show only the ball being tuned + table elements for context
                show_rails = False
                show_pockets = False
                show_cue_line = (self.current_mask_name in ['cue_ball', 'ghost_ball'])
                show_ghost_line = (self.current_mask_name == 'ghost_ball')
        
        # Draw rails
        if show_rails and self.physics.cached_rails:
            for rail_data in self.physics.cached_rails:
                (p1, p2), _ = rail_data
                cv2.line(frame, p1, p2, self.colors['rails'], 1)
        
        # Draw pockets
        if show_pockets and self.physics.cached_pockets:
            for pocket in self.physics.cached_pockets:
                if 'contour' in pocket:
                    # Draw contour if available (live detection)
                    cv2.drawContours(frame, [pocket['contour']], -1, self.colors['pockets'], 2)
                elif 'center' in pocket and 'radius' in pocket:
                    # Draw circle if loaded from cache (no contour available)
                    cv2.circle(frame, pocket['center'], pocket['radius'], self.colors['pockets'], 2)
        
        # Draw ghost line ROI
        if show_ghost_roi and ghost_ball_pos is not None:
            roi_size = self.line_params.get('ghost_line_roi', 65)
            gx, gy = int(ghost_ball_pos[0]), int(ghost_ball_pos[1])
            x1 = max(0, gx - roi_size)
            y1 = max(0, gy - roi_size)
            x2 = min(frame.shape[1], gx + roi_size)
            y2 = min(frame.shape[0], gy + roi_size)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        #Filter detections if in tuning mode
        if self.tuning_mode_active and self.current_mask_name and self.current_mask_name in ['cue_ball', 'ghost_ball', 'solids', 'stripes', 'eight_ball']:
            # Case 1: Tuning a specific ball mask (only draw that ball)
            balls_to_draw = [d for d in detections if d['name'] == self.current_mask_name]
        
        # ADD THIS ELIF BLOCK
        elif self.tuning_mode_active and self.current_mask_name in ['felt', 'rails', 'pockets', 'ghost_line']:
            # Case 2: Tuning a table/line mask (draw NO balls/detections)
            balls_to_draw = []
        
        else:
            # Case 3: Tuning mode is OFF (key '0' - All Masks) or not a known tuning element
            balls_to_draw = detections

        # Draw balls
        for det in balls_to_draw:
            color = self.colors.get(det["name"], (255, 255, 255))
            ball_name = det["name"]
            
            if ball_name in self.ball_params:
                params = self.ball_params[ball_name]
                thickness = params.get('lineThickness', 2)
                # Use the detected radius 'r' from the detection data
                detected_radius = det.get('r', 17) # Use detected radius, with a fallback
                adjusted_r = detected_radius + (thickness // 2) if thickness > 1 else detected_radius + 1
                cv2.circle(frame, (det["x"], det["y"]), adjusted_r, color, thickness)
        
        # Draw cue line
        if show_cue_line and cue_line:
            (x1, y1), (x2, y2) = cue_line
            cv2.line(frame, (x1, y1), (x2, y2), self.colors['cue_line'], 2)
        
        # Draw ghost line trajectory
        if show_ghost_line and ghost_line:
            line_thickness = self.line_params.get('line_thickness', 2)
            for segment in ghost_line:
                (x1, y1), (x2, y2) = segment
                cv2.line(frame, (x1, y1), (x2, y2), self.colors['ghost_line'], line_thickness)
        
        return frame
    
    def run(self):
        print("Billiards Adjuster")
        print("Keys: Q=Quit, O=Toggle Overlays, T=Toggle Bounces")
        print("Keys: S=Save Current Mask, B/P/L=Save Detector Params")
        print("Keys: F=Cache Felt, R=Cache Rails, K=Cache Pockets")
        print("Keys: V=Save Rails to File, N=Save Pockets to File")
        print("Keys: 1-9=Select Mask, 0=All Masks")
        
        fps_counter = 0
        fps_start = time.time()
        displayed_fps = 0
        
        while True:
            loop_start = time.time()
            
            # Capture frame
            img = np.array(self.sct.grab(self.monitor))
            if img is None or img.size == 0:
                continue
            
            # Upload to GPU
            if not self.detector.upload_frame(img):
                continue
            
            # Update trackbar values
            self._get_trackbar_values()

            # --- Main Detection Pipeline ---
            detections, ghost_ball_pos = self.detector.detect_balls()

            cue_line = None  # Cue line detection not implemented
            
            # Use last known stable ghost ball if current is lost
            if ghost_ball_pos is None:
                if self.last_smoothed_ball and self.last_smoothed_ball["name"] == "ghost_ball":
                    ghost_ball_pos = (self.last_smoothed_ball["x"], self.last_smoothed_ball["y"])
            
            # Update last smoothed ball
            for b in detections:
                if b["name"] == "ghost_ball":
                    self.last_smoothed_ball = b
                    break
            
            ghost_line_segments = []
            if ghost_ball_pos is not None:
                cue_ball = next((d for d in detections if d['name'] == 'cue_ball'), None)
                cue_pos = (cue_ball['x'], cue_ball['y']) if cue_ball else None

                direction = self.detector.detect_ghost_line_pixels(ghost_ball_pos)

                if direction is not None:
                    ghost_line_segments = self.physics.calculate_ghost_trajectory(
                        ghost_ball_pos, direction, cue_pos
                    )

            # Update detection counts based on current mask
            if self.current_mask_name in ['cue_ball', 'ghost_ball', 'solids', 'stripes', 'eight_ball']:
                # Count only the currently selected ball type
                self.ball_detection_count = len([d for d in detections if d['name'] == self.current_mask_name])
            elif self.current_mask_name == 'pockets':
                self.ball_detection_count = len(self.physics.cached_pockets) if self.physics.cached_pockets else 0
            elif self.current_mask_name == 'rails':
                self.ball_detection_count = len(self.physics.cached_rails) if self.physics.cached_rails else 0
            elif self.current_mask_name == 'ghost_line':
                self.line_detection_count = len(ghost_line_segments)
            else:
                self.ball_detection_count = len(detections)

            # --- Visualization ---
            display = img.copy()
            display = self.visualize(display, detections, ghost_line_segments, cue_line, ghost_ball_pos)
            
            # --- Mask Preview Window Logic ---
            if self.tuning_mode_active and self.current_mask_name:
                current_mask_pixel_count = 0
                if self.current_mask_name in self.config.masks:
                    vals = self.config.masks[self.current_mask_name]
                    lower = tuple(float(x) for x in vals[:3])
                    upper = tuple(float(x) for x in vals[3:6])
                    
                    gpu_mask = cv2.cuda_GpuMat(self.monitor["height"], self.monitor["width"], cv2.CV_8UC1)  # type: ignore
                    cv2.cuda.inRange(self.detector.gpu_hsv, lower, upper, gpu_mask)
                    mask_cpu = gpu_mask.download()
                    current_mask_pixel_count = cv2.countNonZero(mask_cpu)

                    # --- START: "What You See Is What You Get" Preview Logic ---
                    preview_image = None

                    # Apply margins to mask preview if we're on the rails mask
                    if self.current_mask_name == 'rails':
                        # Apply margins to the mask using the detector's method
                        adjusted_mask = self.detector.apply_rail_margins(mask_cpu)
                        
                        # Perform live rail detection on the adjusted mask
                        self.physics.cached_rails = self.detector.detect_rails_from_mask(adjusted_mask, self.monitor["width"], self.monitor["height"])
                        
                        preview_image = adjusted_mask
                    
                    # Apply morphological operations for felt mask
                    elif self.current_mask_name == 'felt':
                        # Mimic detector's kernel from cache_felt_mask (which uses default 3x3)
                        kernel = np.ones((3,3), np.uint8)
                        mask_cpu = cv2.erode(mask_cpu, kernel, iterations=1)
                        mask_cpu = cv2.dilate(mask_cpu, kernel, iterations=1)

                        # Convert to color to draw play area outline
                        mask_preview = cv2.cvtColor(mask_cpu, cv2.COLOR_GRAY2BGR)

                        # Find contours to show play area boundary
                        contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if contours:
                            # Find largest contour (should be the felt/play area)
                            largest_contour = max(contours, key=cv2.contourArea)

                            # Draw the play area outline in cyan
                            cv2.drawContours(mask_preview, [largest_contour], -1, (255, 255, 0), 3)  # type: ignore

                            # Draw all contours in green for reference
                            cv2.drawContours(mask_preview, contours, -1, (0, 255, 0), 1)  # type: ignore

                        # Use the color preview instead of grayscale
                        preview_image = mask_preview

                    # Apply contour finding for pockets mask
                    elif self.current_mask_name == 'pockets':
                        # Show raw HSV mask + filtered contours visualization
                        # Convert to color to show filtering results
                        mask_preview = cv2.cvtColor(mask_cpu, cv2.COLOR_GRAY2BGR)

                        # Find ALL contours from the raw mask
                        all_contours, _ = cv2.findContours(mask_cpu.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Get filtered pockets that pass detection criteria
                        # This is the single point of live detection for pockets now
                        preview_pockets = self.detector.detect_pockets_from_mask(mask_cpu)
                        
                        # Update the physics engine for the main display
                        self.live_tuning_pockets = preview_pockets
                        self.physics.cached_pockets = self.live_tuning_pockets
                        
                        accepted_contours = [p['contour'] for p in preview_pockets if 'contour' in p]

                        # Draw ALL contours in red (rejected by filters)
                        cv2.drawContours(mask_preview, all_contours, -1, (0, 0, 255), 2)  # type: ignore

                        # Draw ACCEPTED contours in green (passed filters)
                        cv2.drawContours(mask_preview, accepted_contours, -1, (0, 255, 0), 3)  # type: ignore

                        # Use the color preview instead of grayscale
                        preview_image = mask_preview
                    
                    # For balls, show the Canny edge map for live tuning
                    elif self.current_mask_name in ['cue_ball', 'ghost_ball', 'solids', 'stripes', 'eight_ball']:
                        if self.show_canny_preview:
                            # Get the current Canny threshold from the trackbar
                            canny_threshold = max(1, cv2.getTrackbarPos('Ball_Canny', self.window_detector_controls))
                            
                            # Apply Canny edge detection to the HSV mask
                            # The Hough detector's internal Canny uses a high threshold of canny_threshold
                            # and a low threshold of canny_threshold / 2. We replicate that here.
                            canny_output = cv2.Canny(mask_cpu, canny_threshold / 2, canny_threshold)
                            preview_image = canny_output
                        else:
                            # Show the raw HSV mask if Canny preview is off
                            preview_image = mask_cpu
                    else:
                        # For other masks (like ghost_line), show the raw HSV mask
                        preview_image = mask_cpu
                    
                    # Display the final processed preview image
                    cv2.imshow(self.window_hsv_mask, preview_image)
                    # --- END: "What You See Is What You Get" Preview Logic ---
            else:
                # Show a blank image when no mask is selected
                cv2.imshow(self.window_hsv_mask, np.zeros((100, 100), dtype=np.uint8))
            
            # FPS calculation
            if time.time() - fps_start > 1:
                displayed_fps = fps_counter
                fps_counter = 0
                fps_start = time.time()
            
            # Display current mask info
            if self.current_mask_name:
                mask_text = f"Current Mask: {self.current_mask_name.upper().replace('_', ' ')}" #
                (text_width, text_height), baseline = cv2.getTextSize(mask_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2) #
                text_x = (display.shape[1] - text_width) // 2 #
                cv2.putText(display, mask_text, (text_x, 30), #
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

                # Show detection count for current mask
                count_text = ""
                if self.current_mask_name in ['cue_ball', 'ghost_ball', 'solids', 'stripes', 'eight_ball']:
                    count_text = f"Balls: {self.ball_detection_count} | Pixels: {current_mask_pixel_count}"
                elif self.current_mask_name == 'pockets':
                    count_text = f"Detected: {self.ball_detection_count} pocket(s)"
                elif self.current_mask_name == 'rails':
                    count_text = f"Detected: {self.ball_detection_count} rail(s)" # This is len(cached_rails)
                elif self.current_mask_name == 'ghost_line':
                    count_text = f"GL Pixels: {self.detector.last_ghost_line_pixel_count}"

                # Show relevant save key for current mask
                save_key = ""
                if self.current_mask_name in ['cue_ball', 'ghost_ball', 'solids', 'stripes', 'eight_ball']:
                    save_key = "S=Save Mask | B=Save Ball Params | C=Toggle Canny"
                elif self.current_mask_name == 'pockets':
                    save_key = "S=Save Mask | P=Save Pocket Params | K=Cache Pockets"
                elif self.current_mask_name == 'rails':
                    save_key = "S=Save Mask | L=Save Line Params | R=Cache Rails"
                elif self.current_mask_name == 'ghost_line':
                    save_key = "S=Save Mask | L=Save Line Params"
                elif self.current_mask_name == 'felt':
                    save_key = "S=Save Mask | F=Cache Felt"

                if save_key:
                    (text_width, text_height), baseline = cv2.getTextSize(save_key, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2) #
                    text_x = (display.shape[1] - text_width) // 2 #
                    cv2.putText(display, save_key, (text_x, 65), #
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                mask_text = "Current Mask: ALL MASKS" #
                (text_width, text_height), baseline = cv2.getTextSize(mask_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2) #
                text_x = (display.shape[1] - text_width) // 2 #
                cv2.putText(display, mask_text, (text_x, 30), #
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show save cached elements keys
            save_cached_text = "V=Save Rails | N=Save Pockets" #
            (text_width, text_height), baseline = cv2.getTextSize(save_cached_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2) #
            text_x = (display.shape[1] - text_width) // 2 #
            cv2.putText(display, save_cached_text, (text_x, 100), #
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

            # Build and display status text (FPS and detection count)
            status_text = ""
            if displayed_fps > 0:
                status_text = f"FPS: {displayed_fps:.1f}" #
                if self.current_mask_name and count_text: #
                    status_text += f" | {count_text}" #
                (text_width, text_height), baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2) #
                text_x = (display.shape[1] - text_width) // 2 #
                cv2.putText(display, status_text, (text_x, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) #

            # Display main video
            cv2.imshow(self.window_video, display)
            key = cv2.waitKey(1) & 0xFF
            fps_counter += 1
            
            # --- Key Handlers ---
            if key == ord('q'):
                break
            elif key == ord('o'):
                self.show_overlays = not self.show_overlays
                print(f"Overlays: {'ON' if self.show_overlays else 'OFF'}")
            elif key == ord('c'):
                self.show_canny_preview = not self.show_canny_preview
                print(f"Canny Preview: {'ON' if self.show_canny_preview else 'OFF'}")
            elif key == ord('t'):
                self.bounces_enabled = not self.bounces_enabled
                self.physics.set_bounces_enabled(self.bounces_enabled)
                print(f"Bounces: {'ENABLED' if self.bounces_enabled else 'DISABLED'}")
            elif key == ord('s'):
                if self.current_mask_name:
                    self.config.save_mask(self.current_mask_name, self.config.masks[self.current_mask_name])
            elif key == ord('b'):
                # Save ball parameters
                self.config.detector_params['balls'] = self.ball_params
                self.config.save_ball_params(self.ball_params)
            elif key == ord('p'):
                # Save pocket parameters
                self.config.detector_params['pockets'] = self.pocket_params
                self.config.save_pocket_params(self.pocket_params)
            elif key == ord('l'):
                # Save line parameters
                self.config.detector_params['line'] = self.line_params
                self.config.save_line_params(self.line_params)
            elif key == ord('f'):
                if self.detector.cache_felt_mask():
                    print("✓ Cached felt mask to GPU")
            elif key == ord('r'):
                # Re-run rail detection using the current mask & params
                if "rails" in self.config.masks:
                    vals = self.config.masks["rails"]
                    lower = tuple(float(x) for x in vals[:3])
                    upper = tuple(float(x) for x in vals[3:6])
                    
                    # Perform masking on CPU
                    frame_bgr = self.detector.get_frame()
                    hsv_cpu = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                    mask_cpu = cv2.inRange(hsv_cpu, np.array(lower), np.array(upper))

                    adjusted_mask = self.detector.apply_rail_margins(mask_cpu)
                    self.physics.cached_rails = self.detector.detect_rails_from_mask(adjusted_mask, self.monitor["width"], self.monitor["height"])
                    print(f"✓ Cached {len(self.physics.cached_rails)} rails")
            elif key == ord('k'):
                # Re-run pocket detection using the current mask & params
                if "pockets" in self.config.masks:
                    vals = self.config.masks["pockets"]
                    lower = tuple(float(x) for x in vals[:3])
                    upper = tuple(float(x) for x in vals[3:6])

                    # Perform masking on CPU
                    frame_bgr = self.detector.get_frame()
                    hsv_cpu = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                    mask_cpu = cv2.inRange(hsv_cpu, np.array(lower), np.array(upper))

                    pockets = self.detector.detect_pockets_from_mask(mask_cpu)
                    self.physics.cached_pockets = pockets
                    print(f"✓ Cached {len(pockets)} pockets")
            elif key == ord('v'):
                # Save rails only
                self.config.save_cached_elements(rails=self.physics.cached_rails)
            elif key == ord('n'):
                # Save pockets only
                self.config.save_cached_elements(pockets=self.physics.cached_pockets)
            elif key in self.mask_map:
                new_mask = self.mask_map[key]
                if new_mask != self.current_mask_name:
                    self.current_mask_name = new_mask
                    self.tuning_mode_active = (new_mask is not None)
                    self._update_trackbar_positions()
            
            # Frame timing
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.frame_time - elapsed)
            time.sleep(sleep_time)
        
        cv2.destroyAllWindows()
        print("Adjuster stopped")

if __name__ == "__main__":
    try:
        adjuster = BilliardsAdjuster()
        adjuster.run()
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
