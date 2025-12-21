import math
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any

class BilliardsPhysics:
    def __init__(self, monitor: Dict[str, int], cached_rails: Optional[List] = None,
                 cached_pockets: Optional[List] = None, max_bounces: int = 2):
        self.monitor = monitor
        self.cached_rails = cached_rails
        self.cached_pockets = cached_pockets
        self.max_bounces = max_bounces
    
    def calculate_ghost_trajectory(self, ghost_center: Tuple[int, int], 
                                   direction: Tuple[float, float],
                                   cue_center: Optional[Tuple[int, int]] = None) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        
        gx, gy = ghost_center
        vx, vy = direction
        
        # Flip direction if needed to point away from cue ball
        if cue_center:
            cx, cy = cue_center
            vec_ghost_to_cue = (cx - gx, cy - gy)
            mag = math.hypot(vec_ghost_to_cue[0], vec_ghost_to_cue[1])
            if mag > 0:
                norm_ghost_to_cue = (vec_ghost_to_cue[0] / mag, vec_ghost_to_cue[1] / mag)
                desired_vx, desired_vy = -norm_ghost_to_cue[0], -norm_ghost_to_cue[1]
                dot_product = vx * desired_vx + vy * desired_vy
                if dot_product < 0:
                    vx, vy = -vx, -vy
        
        # Normalize direction
        magnitude = math.hypot(vx, vy)
        if magnitude == 0:
            return []
        vx_norm, vy_norm = vx / magnitude, vy / magnitude
        
        current_pos = (int(gx), int(gy))
        current_dir = (vx_norm, vy_norm)
        trajectory_segments = []
        hit_rail_normal = None
        
        for bounce_count in range(self.max_bounces + 1):
            # Extend ray far out
            length = 3000
            p_end_long = (int(current_pos[0] + current_dir[0] * length),
                         int(current_pos[1] + current_dir[1] * length))
            
            closest_intersection_point = None
            closest_intersection_type = None
            min_dist_sq = float('inf')
            
            # Check pocket intersections (optimized with early rejection)
            if self.cached_pockets:
                for pocket_data in self.cached_pockets:
                    pocket_cx, pocket_cy = pocket_data['center']
                    pocket_radius = pocket_data['radius']
                    pocket_contour = pocket_data.get('contour')  # May be None if loaded from cache

                    # Early rejection: calculate closest approach to pocket
                    dx = p_end_long[0] - current_pos[0]
                    dy = p_end_long[1] - current_pos[1]
                    ray_len_sq = dx*dx + dy*dy

                    if ray_len_sq < 1:
                        continue

                    # Project pocket center onto ray
                    to_pocket_x = pocket_cx - current_pos[0]
                    to_pocket_y = pocket_cy - current_pos[1]
                    t_proj = (to_pocket_x * dx + to_pocket_y * dy) / ray_len_sq

                    # Skip if pocket behind ray start or beyond ray end
                    if t_proj < 0 or t_proj > 1:
                        continue

                    # Closest point on ray to pocket
                    closest_x = current_pos[0] + t_proj * dx
                    closest_y = current_pos[1] + t_proj * dy
                    closest_dist = math.hypot(closest_x - pocket_cx, closest_y - pocket_cy)

                    # Skip if ray doesn't get within pocket radius
                    if closest_dist > pocket_radius * 1.5:
                        continue

                    # Fine sampling (optimized: 50 instead of 500)
                    num_samples = 50
                    for i in range(num_samples):
                        t = i / float(num_samples)
                        px = int(current_pos[0] + t * dx)
                        py = int(current_pos[1] + t * dy)

                        dist_to_center = math.hypot(px - pocket_cx, py - pocket_cy)
                        if dist_to_center < pocket_radius * 1.2:
                            # Check if point is inside pocket (use contour if available, else circle)
                            is_inside = False
                            if pocket_contour is not None:
                                is_inside = cv2.pointPolygonTest(pocket_contour, (px, py), False) >= 0
                            else:
                                # Fallback: use circular approximation if no contour available
                                is_inside = dist_to_center <= pocket_radius

                            if is_inside:
                                dist_from_start_sq = (px - current_pos[0])**2 + (py - current_pos[1])**2
                                if dist_from_start_sq > 25 and dist_from_start_sq < min_dist_sq:
                                    min_dist_sq = dist_from_start_sq
                                    closest_intersection_point = (px, py)
                                    closest_intersection_type = 'pocket'
                                break
            
            # Check rail intersections
            if self.cached_rails and closest_intersection_type != 'pocket':
                for (rail_p1, rail_p2), rail_normal_vec in self.cached_rails:
                    intersection_point = self._get_line_intersection(current_pos, p_end_long, rail_p1, rail_p2)
                    if intersection_point:
                        dist_from_start_sq = (intersection_point[0] - current_pos[0])**2 + (intersection_point[1] - current_pos[1])**2
                        if dist_from_start_sq > 25 and dist_from_start_sq < min_dist_sq:
                            min_dist_sq = dist_from_start_sq
                            closest_intersection_point = intersection_point
                            closest_intersection_type = 'rail'
                            hit_rail_normal = rail_normal_vec
            
            # Add segment to trajectory
            if closest_intersection_point:
                trajectory_segments.append((current_pos, closest_intersection_point))
                current_pos = closest_intersection_point
                
                if closest_intersection_type == 'pocket':
                    break  # Ball goes in pocket, trajectory ends
                elif closest_intersection_type == 'rail':
                    # Calculate reflection
                    v_in_norm = current_dir
                    
                    if hit_rail_normal is not None:
                        # Flip normal if needed
                        dot_product_check = v_in_norm[0] * hit_rail_normal[0] + v_in_norm[1] * hit_rail_normal[1]
                        effective_rail_normal = (-hit_rail_normal[0], -hit_rail_normal[1]) if dot_product_check > 0 else hit_rail_normal
                        
                        # Reflect velocity
                        dot_product = v_in_norm[0] * effective_rail_normal[0] + v_in_norm[1] * effective_rail_normal[1]
                        v_reflect_norm = (v_in_norm[0] - 2 * dot_product * effective_rail_normal[0],
                                         v_in_norm[1] - 2 * dot_product * effective_rail_normal[1])
                        current_dir = v_reflect_norm
            else:
                # No intersection found, clip to screen bounds
                clipped_line = self._clip_line(current_pos, p_end_long)
                if clipped_line:
                    trajectory_segments.append(clipped_line)
                break
        
        return trajectory_segments
    
    def _get_line_intersection(self, p1: Tuple[int, int], p2: Tuple[int, int],
                               p3: Tuple[int, int], p4: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Calculate intersection point of two line segments"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None
        
        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
        
        t = t_num / den
        u = u_num / den
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            return (int(px), int(py))
        
        return None
    
    def _clip_line(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Clip line to screen bounds using Liang-Barsky algorithm"""
        xmin, ymin = 0, 0
        xmax, ymax = self.monitor["width"], self.monitor["height"]
        x1, y1 = pt1
        x2, y2 = pt2
        dx, dy = x2 - x1, y2 - y1
        t0, t1 = 0.0, 1.0
        
        def clip(p, q):
            nonlocal t0, t1
            if p == 0:
                return q >= 0
            t = q / p
            if p < 0:
                if t > t1:
                    return False
                if t > t0:
                    t0 = t
            else:
                if t < t0:
                    return False
                if t < t1:
                    t1 = t
            return True
        
        if (clip(-dx, x1 - xmin) and clip(dx, xmax - x1)
                and clip(-dy, y1 - ymin) and clip(dy, ymax - y1)):
            nx1 = x1 + t0 * dx
            ny1 = y1 + t0 * dy
            nx2 = x1 + t1 * dx
            ny2 = y1 + t1 * dy
            return (int(nx1), int(ny1)), (int(nx2), int(ny2))
        return None
    
    def set_bounces_enabled(self, enabled: bool):
        """Enable/disable bounce calculations"""
        self.max_bounces = 2 if enabled else 0
