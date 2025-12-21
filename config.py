import json
from typing import Dict, Any, List, Tuple, Optional

class BilliardsConfig:
    def __init__(self, config_file: str = 'billiards_calibration.json'):
        self.config_file = config_file
        self.masks: Dict[str, List[int]] = {}
        self.detector_params: Dict[str, Any] = {}
        self.cached_elements: Dict[str, Any] = {}
        
    def load(self) -> bool:
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                
                # Convert dict-style masks to list-style if necessary
                loaded_masks = data.get('masks', {})
                for name, values in loaded_masks.items():
                    if isinstance(values, dict): # Handle dict format from older configs
                        self.masks[name] = [values.get(k, 0) for k in ['h_min', 's_min', 'v_min', 'h_max', 's_max', 'v_max']]
                    elif isinstance(values, list): # Handle list format
                        self.masks[name] = values
                
                self.detector_params = data.get('detector_params', {})
                self.cached_elements = data.get('cached_elements', {})
                print(f"✓ Loaded {self.config_file}")
                return True
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {self.config_file}")
            return False
        except Exception as e:
            print(f"✗ Error loading config: {e}")
            return False
    
    def save_mask(self, mask_name: str, hsv_values: List[int]) -> bool:
        """Save a single mask's HSV values"""
        try:
            data = self._load_full_config()
            if 'masks' not in data:
                data['masks'] = {}
            
            # Update the specific mask value in the loaded data
            data['masks'][mask_name] = hsv_values
            self._write_config(data)
            print(f"✓ Saved {mask_name} mask")
            return True
        except Exception as e:
            print(f"✗ Error saving mask: {e}")
            return False
    
    def save_ball_params(self, ball_params: Dict[str, Dict[str, Any]]) -> bool:
        """Save ball detector parameters"""
        try:
            data = self._load_full_config()
            if 'detector_params' not in data:
                data['detector_params'] = {}
            data['detector_params']['balls'] = ball_params
            
            self.detector_params = data['detector_params'] # Update internal state
            self._write_config(data)
            print(f"✓ Saved ball detector params")
            return True
        except Exception as e:
            print(f"✗ Error saving ball params: {e}")
            return False
    
    def save_pocket_params(self, pocket_params: Dict[str, Any]) -> bool:
        """Save pocket detector parameters"""
        try:
            data = self._load_full_config()
            if 'detector_params' not in data:
                data['detector_params'] = {}
            data['detector_params']['pockets'] = pocket_params

            self.detector_params = data['detector_params'] # Update internal state
            self._write_config(data)
            print(f"✓ Saved pocket detector params")
            return True
        except Exception as e:
            print(f"✗ Error saving pocket params: {e}")
            return False
    
    def save_line_params(self, line_params: Dict[str, Any]) -> bool:
        """Save line detector parameters"""
        try:
            data = self._load_full_config()
            if 'detector_params' not in data:
                data['detector_params'] = {}
            data['detector_params']['line'] = line_params
            
            self.detector_params = data['detector_params'] # Update internal state
            self._write_config(data)
            print(f"✓ Saved line detector params")
            return True
        except Exception as e:
            print(f"✗ Error saving line params: {e}")
            return False
    
    def save_cached_elements(self, rails: Optional[List] = None, 
                            pockets: Optional[List] = None) -> bool:
        """Save cached static elements (rails, pockets)"""
        try:
            data = self._load_full_config()
            data.setdefault('cached_elements', {})
            
            if rails is not None:
                serializable_rails = [
                    [[list(p1), list(p2)], list(normal)]
                    for (p1, p2), normal in rails
                ]
                data['cached_elements']['rails'] = serializable_rails
                print("✓ Cached rails saved")
            
            if pockets is not None:
                serializable_pockets = [
                    {'center': list(p['center']), 'radius': p['radius']}
                    for p in pockets if 'center' in p and 'radius' in p
                ]
                data['cached_elements']['pockets'] = serializable_pockets
                print("✓ Cached pockets saved")

            self.cached_elements = data['cached_elements']
            self._write_config(data)
            return True
        except Exception as e:
            print(f"✗ Error saving cached elements: {e}")
            return False
    
    def get_cached_rails(self) -> Optional[List]:
        """Get cached rails with proper tuple conversion"""
        loaded_rails = self.cached_elements.get('rails')
        if loaded_rails:
            return [
                ((tuple(p1), tuple(p2)), tuple(normal))
                for (p1, p2), normal in loaded_rails
            ]
        return None
    
    def get_cached_pockets(self) -> Optional[List]:
        """Get cached pockets"""
        return self.cached_elements.get('pockets')
    
    def _load_full_config(self) -> Dict:
        """Load full config or return empty dict"""
        try:
            with open(self.config_file, 'r') as f:
                content = f.read()
                if not content: # Handle empty file
                    return {}
                return json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _write_config(self, data: Dict) -> None:
        """Write config to file"""
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=4)

def get_default_ball_params() -> Dict[str, Any]:
    """Default ball detector parameters"""
    return {
        'minDist': 20,
        'cannyThreshold': 60,
        'votesThreshold': 8,
        'minRadius': 12,
        'maxRadius': 17,
        'lineThickness': 2
    }

def get_default_pocket_params() -> Dict[str, Any]:
    """Default pocket detector parameters"""
    return {
        'minDist': 50,
        'cannyThreshold': 100,
        'votesThreshold': 20,
        'minRadius': 20,
        'maxRadius': 40,
        'min_area': 500,
        'min_circularity': 70,
        'max_area': 5000
    }

def get_default_line_params() -> Dict[str, Any]:
    """Default line detector parameters"""
    return {
        'angle_tolerance': 10,
        'edge_margin_left': 0,
        'edge_margin_right': 0,
        'edge_margin_top_left': 0,
        'edge_margin_top_right': 0,
        'edge_margin_bottom_left': 0,
        'edge_margin_bottom_right': 0,
        'line_thickness': 2,
        'ghost_line_roi': 65,
        'merge_tolerance_px': 10,
        'merge_tolerance_angle': 5,
        'rail_line_shorten_pixels': 0,
        'rail_offset_pixels': 0,
        'center_rail_extension': 0,
        'max_rail_gap': 10
    }
