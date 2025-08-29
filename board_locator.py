import os
import time
import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
from typing import Dict, Tuple, Optional
# cairosvg is no longer needed for this class
# import cairosvg 

class BoardLocator:
    """
    Finds the 4-player chessboard on screen by locating the four kings,
    determining the board's orientation, and calculating a precise vector-based
    coordinate system for the squares.
    This version uses pre-rendered PNG templates for the kings.
    """
    def __init__(self, king_png_path: str, window_title: str, confidence: float = 0.95):
        self.king_path = king_png_path
        self.window_title = window_title # Note: This is now just for reference/logging
        self.confidence = confidence
        self.king_templates = self._load_king_template_pairs()

    def _load_king_template_pairs(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Loads the eight king PNGs (light and dark versions for each color)
        into OpenCV format with transparency.
        """
        templates = {}
        king_map = {
            'R': 'r', # Maps internal color code to filename prefix
            'B': 'b',
            'Y': 'y',
            'G': 'g',
        }
        print("Loading king templates from PNG files...")
        for color_code, prefix in king_map.items():
            light_path = os.path.join(self.king_path, f"{prefix}k_light.png")
            dark_path = os.path.join(self.king_path, f"{prefix}k_dark.png")

            # Check if files exist before trying to load
            if not os.path.exists(light_path):
                raise FileNotFoundError(f"King template not found at {light_path}.")
            if not os.path.exists(dark_path):
                raise FileNotFoundError(f"King template not found at {dark_path}.")

            # Load images with alpha channel
            light_img = cv2.imread(light_path, cv2.IMREAD_UNCHANGED)
            dark_img = cv2.imread(dark_path, cv2.IMREAD_UNCHANGED)

            if light_img is None or dark_img is None:
                raise IOError(f"Could not read image files for {color_code} King.")
            
            templates[color_code] = {'light': light_img, 'dark': dark_img}
            print(f"  - Loaded {color_code} King templates (light and dark).")
        return templates

    def _find_piece_center(self, screenshot: np.ndarray, template: np.ndarray) -> Optional[Tuple[int, int]]:
        """Finds the center of a given piece template within a screenshot."""
        # Ensure template has an alpha channel for masking
        if template.shape[2] < 4:
            print(f"Warning: Template is missing alpha channel. Results may be inaccurate.")
            # Fallback to simple template matching if no alpha
            screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        else:
            # Use alpha mask for more precise matching
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            template_bgr = template[:, :, :3]
            alpha_mask = template[:, :, 3]
            res = cv2.matchTemplate(screenshot_bgr, template_bgr, cv2.TM_CCORR_NORMED, mask=alpha_mask)

        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= self.confidence:
            h, w = template.shape[:2]
            return (max_loc[0] + w // 2, max_loc[1] + h // 2)
        return None

    def calibrate(self, setup: str = 'modern') -> Optional[Dict]:
        """
        Main public method. Finds kings on the full screen using a two-step
        (light/dark) search, calculates geometry, and performs a sanity check.
        NOTE: Assumes the correct window has ALREADY been brought to the foreground by the caller.
        """
        print("Taking a screenshot of the entire screen...")
        try:
            screenshot = pyautogui.screenshot()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGRA)
        except Exception as e:
            print(f"Error capturing full-screen screenshot: {e}")
            return None

        print("Searching for kings on screen...")
        king_coords_raw = {}
        for color, template_pair in self.king_templates.items():
            coords = None
            
            # 1. Try to find the light version first
            print(f"  - Searching for {color} King (light version)...")
            coords = self._find_piece_center(screenshot_cv, template_pair['light'])

            # 2. If not found, try the dark version
            if coords is None:
                print(f"    - Light version not found. Trying dark version...")
                coords = self._find_piece_center(screenshot_cv, template_pair['dark'])
            
            # 3. Check the final result and store it or fail
            if coords:
                king_coords_raw[color] = np.array(coords, dtype=float)
                print(f"  - Found {color} King at {coords}")
            else:
                print(f"  - FAILED to find {color} King (both light and dark versions). Calibration aborted.")
                print(f"    Suggestion: Ensure the game window is fully visible and not obscured.")
                return None
        
        if len(king_coords_raw) != 4:
            # This check is somewhat redundant now but good for safety
            print("ERROR: Could not locate all four kings. Is it the start of the game?")
            return None
        
        print(f"\nAll kings found. Calculating board geometry for '{setup}' setup using vector math...")
        return self._calculate_geometry(king_coords_raw, setup)

    def _calculate_geometry(self, king_coords: Dict[str, np.ndarray], setup: str) -> Optional[Dict]:
        """
        Uses vector math based on king positions to derive a universal board coordinate system.
        The algebraic vectors and king positions change based on the game setup.
        """
        vec_ry = king_coords['Y'] - king_coords['R']
        vec_bg = king_coords['G'] - king_coords['B']
        
        # Define algebraic king positions (col, row) and the resulting matrix A for each setup
        if setup == 'classic':
            # R=h1(7,0), Y=g14(6,13) -> vec_ry = (-1, 13)
            # B=a8(0,7), G=n7(13,6) -> vec_bg = (13, -1)
            A = np.array([[-1, 13], [13, -1]])
            king_alg = {'R': (7, 0), 'Y': (6, 13), 'B': (0, 7), 'G': (13, 6)}
        else: # Default to 'modern'
            # R=h1(7,0), Y=g14(6,13) -> vec_ry = (-1, 13)
            # B=a7(0,6), G=n8(13,7) -> vec_bg = (13, 1)
            A = np.array([[-1, 13], [13, 1]])
            king_alg = {'R': (7, 0), 'Y': (6, 13), 'B': (0, 6), 'G': (13, 7)}
        
        if np.linalg.det(A) == 0:
            print(f"!!! CRITICAL ERROR: Geometry matrix for setup '{setup}' is singular. Cannot calculate vectors.")
            return None
            
        A_inv = np.linalg.inv(A)
        
        # Matrix of the two displacement vectors stacked as rows
        vec_matrix_rows = np.vstack([vec_ry, vec_bg])
        
        # Solve the system of linear equations: U = A_inv * V
        # The result is a 2x2 matrix where row 0 is U_col and row 1 is U_row
        unit_vectors = np.dot(A_inv, vec_matrix_rows)
        unit_vec_col, unit_vec_row = unit_vectors

        square_size = (np.linalg.norm(unit_vec_col) + np.linalg.norm(unit_vec_row)) / 2.0
        print(f"Calculated average square size: {square_size:.2f} pixels")
        
        print(f"Unit vector for 1 column (e.g., a->b): {unit_vec_col.round(2)}")
        print(f"Unit vector for 1 row (e.g., 1->2):   {unit_vec_row.round(2)}")

        # Origin is calculated from the Red King's position, which is the same in both setups
        origin_a1_center_px = king_coords['R'] - (7 * unit_vec_col)
        print(f"Calculated origin (center of a1): {origin_a1_center_px.round(1)}")

        print("\nPerforming sanity check by re-calculating king positions...")
        errors = {}
        # The correct king_alg dictionary is used here based on the setup
        for color, (col_offset, row_offset) in king_alg.items():
            calculated_pos = origin_a1_center_px + col_offset * unit_vec_col + row_offset * unit_vec_row
            found_pos = king_coords[color]
            error_dist = np.linalg.norm(calculated_pos - found_pos)
            errors[color] = error_dist
            print(f"  - {color} King: Found at {found_pos.astype(int)}, Calculated at {calculated_pos.astype(int)}. Error: {error_dist:.2f} pixels.")
        
        max_error = max(errors.values())
        if max_error > square_size / 3:
            print(f"!!! CRITICAL VALIDATION FAILED: Max error ({max_error:.1f}px) is too large. Calibration is unreliable.")
            return None
            
        print("Sanity check passed. Calibration is valid.")
        
        return {
            'origin_a1_center_px': origin_a1_center_px,
            'unit_vec_col': unit_vec_col,
            'unit_vec_row': unit_vec_row,
            'square_size': square_size
        }