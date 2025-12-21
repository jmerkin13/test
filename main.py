import cv2
import mss
import numpy as np
import time
from typing import Dict, Tuple, Any, List
from ultralytics import YOLO

from config import BilliardsConfig, get_default_line_params
from logic import BilliardsPhysics

class BilliardsRuntime:
    def __init__(self, config_file: str = 'billiards_calibration.json', 
                 model_path: str = '/home/justin/Pictures/copy of project/best.pt'):
        # Load config
        self.config = BilliardsConfig(config_file)
        if not self.config.load():
            print("âš ï¸ Using default parameters")
        
        # Screen capture
        self.sct = mss.mss()
        self.monitor = {"left": 2171, "top": 124, "width": 1325, "height": 755}
        
        # Load YOLO model with GPU
        self.model = YOLO(model_path)
        self.device = 'cuda:0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        self.class_names = ['eight_ball', 'cue_ball', 'ghost_ball', 'solids', 'stripes']
        
        # Warm up model
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, device=self.device, verbose=False)
        
        print(f"âœ“ Loaded YOLO model: {model_path}")
        print(f"âœ“ Using device: {self.device}")
        
        # Setup line parameters
        self.line_params = self.config.detector_params.get('line', get_default_line_params())
        
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
        
        # State
        self.show_overlays = True
        self.bounces_enabled = True
        self.ghost_line_only = False
        self.detect_only_cue_ghost = False  # New toggle
        
        # Cached detections for static balls
        self.cached_static_balls = []
        self.static_ball_update_counter = 0
        self.static_ball_update_interval = 30  # Update every 30 frames
        
        # Colors
        self.colors = {
            'cue_ball': (255, 255, 255), 'ghost_ball': (0, 255, 0),
            'solids': (0, 0, 255), 'stripes': (255, 0, 0),
            'eight_ball': (0, 0, 0), 'pockets': (255, 255, 0),
            'rails': (0, 255, 255), 'ghost_line': (255, 0, 255),
            'cue_line': (128, 128, 128)
        }
        
        # Performance
        self.target_fps = 45
        self.frame_time = 1.0 / self.target_fps
        
        # Window
        self.window_name = "ðŸŽ± Billiards Runtime (YOLO)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, 909, 554)
    
    def detect_balls(self, frame: np.ndarray, detect_all: bool = False) -> Tuple[List[Dict], Tuple[int, int]]:
        """Detect balls using YOLO"""
        if detect_all:
            # Full detection
            results = self.model(frame, device=self.device, verbose=False, imgsz=416, half=True)[0]
            
            detections = []
            ghost_ball_pos = None
            static_balls = []
            
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                ball_name = self.class_names[cls_id]
                
                detection = {
                    'name': ball_name,
                    'x': center_x,
                    'y': center_y,
                    'conf': conf
                }
                
                detections.append(detection)
                
                if ball_name == 'ghost_ball':
                    ghost_ball_pos = (center_x, center_y)
                elif ball_name not in ['cue_ball', 'ghost_ball']:
                    static_balls.append(detection)
            
            # Cache static balls only if ghost ball was found
            if ghost_ball_pos:
                self.cached_static_balls = static_balls
            
            return detections, ghost_ball_pos
        else:
            # Fast detection - only cue_ball and ghost_ball
            results = self.model(frame, device=self.device, verbose=False, imgsz=416, half=True, classes=[1, 2])[0]
            
            detections = []
            ghost_ball_pos = None
            
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                ball_name = self.class_names[cls_id]
                
                detections.append({
                    'name': ball_name,
                    'x': center_x,
                    'y': center_y,
                    'conf': conf
                })
                
                if ball_name == 'ghost_ball':
                    ghost_ball_pos = (center_x, center_y)
            
            # Only add cached static balls if ghost ball is present
            if ghost_ball_pos:
                detections.extend(self.cached_static_balls)
            
            return detections, ghost_ball_pos
    
    def detect_ghost_line_direction(self, frame: np.ndarray, ghost_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Detect ghost line direction using HSV masking (existing method)"""
        gx, gy = ghost_pos
        roi_size = self.line_params.get('ghost_line_roi', 65)
        
        x1 = max(0, gx - roi_size)
        y1 = max(0, gy - roi_size)
        x2 = min(frame.shape[1], gx + roi_size)
        y2 = min(frame.shape[0], gy + roi_size)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # HSV mask for ghost line
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ghost_line_mask = self.config.masks.get('ghost_line', [0, 0, 204, 179, 124, 255])
        lower = np.array(ghost_line_mask[:3], dtype=np.uint8)
        upper = np.array(ghost_line_mask[3:], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 10:
            return None
        
        # Fit line
        [vx, vy, x0, y0] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        return (float(vx[0]), float(vy[0]))
    
    def visualize(self, frame: np.ndarray, detections, ghost_line, cue_line, cached_rails, cached_pockets) -> np.ndarray:
        """Draw overlays on frame"""
        if not self.show_overlays:
            return frame

        # Ghost line only mode
        if self.ghost_line_only:
            if ghost_line:
                for segment in ghost_line:
                    (x1, y1), (x2, y2) = segment
                    cv2.line(frame, (x1, y1), (x2, y2), self.colors['ghost_line'], 2)
            return frame

        # Draw rails
        if cached_rails:
            for rail_data in cached_rails:
                (p1, p2), _ = rail_data
                cv2.line(frame, p1, p2, self.colors['rails'], 2)

        # Draw pockets
        if cached_pockets:
            for pocket in cached_pockets:
                if 'center' in pocket and 'radius' in pocket:
                    cv2.circle(frame, tuple(pocket['center']), pocket['radius'], self.colors['pockets'], 2)

        # Draw balls
        for det in detections:
            color = self.colors.get(det["name"], (255, 255, 255))
            cv2.circle(frame, (det["x"], det["y"]), 17, color, 2)

        # Draw cue line
        if cue_line:
            (x1, y1), (x2, y2) = cue_line
            cv2.line(frame, (x1, y1), (x2, y2), self.colors['cue_line'], 2)

        # Draw ghost line trajectory
        if ghost_line:
            for segment in ghost_line:
                (x1, y1), (x2, y2) = segment
                cv2.line(frame, (x1, y1), (x2, y2), self.colors['ghost_line'], 2)

        return frame
    
    def run(self):
        """Main runtime loop"""
        print("ðŸš€ Billiards Runtime (YOLO)")
        print("Keys: Q=Quit, O=Toggle Overlays, T=Toggle Bounces, G=Ghost Line Only, C=Cue+Ghost Only")
        
        fps_counter = 0
        fps_start = time.time()
        displayed_fps = 0
        
        # Profiling
        times = {'capture': [], 'yolo': [], 'ghost_line': [], 'physics': [], 'viz': []}
        
        while True:
            loop_start = time.time()
            
            # Capture frame
            t0 = time.time()
            img = np.array(self.sct.grab(self.monitor))
            if img is None or img.size == 0:
                continue
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            times['capture'].append(time.time() - t0)
            
            # Detect balls with YOLO
            t0 = time.time()
            if self.detect_only_cue_ghost:
                # Always only detect cue + ghost when toggle is on
                detections, ghost_ball_pos = self.detect_balls(frame, detect_all=False)
            else:
                # Use interval-based full detection
                self.static_ball_update_counter += 1
                detect_all = self.static_ball_update_counter >= self.static_ball_update_interval
                if detect_all:
                    self.static_ball_update_counter = 0
                detections, ghost_ball_pos = self.detect_balls(frame, detect_all=detect_all)
            times['yolo'].append(time.time() - t0)
            
            # Calculate trajectories
            ghost_line = None
            cue_line = None
            
            if ghost_ball_pos:
                cue_ball = next((d for d in detections if d["name"] == "cue_ball"), None)
                
                if cue_ball:
                    cue_line = (
                        (cue_ball["x"], cue_ball["y"]),
                        (ghost_ball_pos[0], ghost_ball_pos[1])
                    )
                
                # Detect ghost line direction
                t0 = time.time()
                direction = self.detect_ghost_line_direction(frame, ghost_ball_pos)
                times['ghost_line'].append(time.time() - t0)
                
                if direction is not None:
                    t0 = time.time()
                    cue_center = (cue_ball['x'], cue_ball['y']) if cue_ball else None
                    ghost_line = self.physics.calculate_ghost_trajectory(
                        ghost_ball_pos, direction, cue_center
                    )
                    times['physics'].append(time.time() - t0)
            
            # Visualize
            t0 = time.time()
            display = self.visualize(
                frame, detections, ghost_line, cue_line,
                self.physics.cached_rails, self.physics.cached_pockets
            )
            times['viz'].append(time.time() - t0)
            
            # FPS counter
            fps_elapsed = time.time() - fps_start
            if fps_elapsed > 1.0:
                displayed_fps = fps_counter / fps_elapsed
                fps_counter = 0
                fps_start = time.time()
                
                # Print profiling stats
                print("\n=== Profiling ===")
                for key in times:
                    if times[key]:
                        avg = sum(times[key]) / len(times[key]) * 1000
                        print(f"{key:12s}: {avg:6.2f}ms")
                print(f"FPS: {displayed_fps:.1f}")
                times = {'capture': [], 'yolo': [], 'ghost_line': [], 'physics': [], 'viz': []}
            
            if displayed_fps > 0:
                cv2.putText(display, f"FPS: {displayed_fps:.1f}", (10, display.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            fps_counter += 1
            
            # Handle keys
            if key == ord('q'):
                break
            elif key == ord('o'):
                self.show_overlays = not self.show_overlays
                print(f"Overlays: {'ON' if self.show_overlays else 'OFF'}")
            elif key == ord('t'):
                self.bounces_enabled = not self.bounces_enabled
                self.physics.set_bounces_enabled(self.bounces_enabled)
                print(f"Bounces: {'ENABLED' if self.bounces_enabled else 'DISABLED'}")
            elif key == ord('g'):
                self.ghost_line_only = not self.ghost_line_only
                print(f"Ghost Line Only: {'ON' if self.ghost_line_only else 'OFF'}")
            elif key == ord('c'):
                self.detect_only_cue_ghost = not self.detect_only_cue_ghost
                print(f"Detect Only Cue+Ghost: {'ON' if self.detect_only_cue_ghost else 'OFF'}")
            
            # Frame timing
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.frame_time - elapsed)
            time.sleep(sleep_time)
        
        cv2.destroyAllWindows()
        print("ðŸ‘‹ Runtime stopped")

if __name__ == "__main__":
    try:
        runtime = BilliardsRuntime()
        runtime.run()
    except Exception as e:
        import traceback
        print(f"âœ— Error: {e}")
        traceback.print_exc()
