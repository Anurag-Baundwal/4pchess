# game_controller.py
# v28 (Adds OCR-based chat functionality to say "gg" at game end)
import os
import sys
import time
import threading
import argparse
import json
from typing import Dict, Optional, Tuple, List, Set, Union
import numpy as np

# Imports for launching the fetcher
import platform
import subprocess
import shlex

import pyautogui
import pygetwindow as gw

# --- NEW IMPORTS ---
try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("Error: The 'pytesseract' or 'Pillow' library is not installed.")
    print("Please install them by running this command in your terminal:")
    print("pip install pytesseract Pillow")
    sys.exit(1)
# --------------------

try:
    import win32gui
    import win32con
    import win32process
    import win32api
except ImportError:
    print("Error: The 'pywin32' library is not installed and is required for robust window switching.")
    print("Please install it by running this command in your terminal:")
    print("pip install pywin32")
    sys.exit(1)

import uci_wrapper
from board_locator import BoardLocator
import tablebase

# --- CONFIGURATION ---
WINDOW_CONFIG = {
    'BOT_1_TITLE': "Bot-1",
    'BOT_2_TITLE': "Bot-2",
}
PIECE_ASSET_PATH = os.path.join('assets', 'pieces_svg')
BOARD_DIMENSIONS = 14
GAME_STATE_FILE = "game_state.json"
EVAL_LOG_FILE = "eval_log.txt"
POLL_INTERVAL_SECONDS = 0.1

# Time Management Constants (for dynamic TC)
MAX_MOVE_MS = 30000
MIN_REMAINING_MOVE_MS = 30000
MIN_MOVE_TIME_MS = 100
TIME_DIVISOR = 20.0
SAFETY_MARGIN = 0.90
# ---------------------

# --- NEW: If Tesseract is not in your PATH, uncomment and set the path here ---
# For example, on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# -----------------------------------------------------------------------------

def launch_move_fetcher(url: str):
    """Launches move_fetcher.py in a separate terminal based on the OS."""
    current_os = platform.system()
    print(f"[{current_os.upper()}] Launching move fetcher in a new terminal...")
    fetcher_command_args = [sys.executable, "move_fetcher.py", "--url", url]
    try:
        if current_os == "Windows":
            subprocess.Popen(['start', 'cmd', '/k'] + fetcher_command_args, shell=True)
        elif current_os == "Darwin": # macOS
            fetcher_script_cmd = f"cd {shlex.quote(os.getcwd())}; {shlex.join(fetcher_command_args)}"
            subprocess.run(['osascript', '-e', f'tell app "Terminal" to do script "{fetcher_script_cmd}"'], check=True)
        elif current_os == "Linux":
            try:
                subprocess.Popen(['gnome-terminal', '--'] + fetcher_command_args)
            except FileNotFoundError:
                print("\nPlease run this command manually in a separate terminal:")
                print(f"  {shlex.join(fetcher_command_args)}")
        else:
            print(f"Unsupported OS: {current_os}. Please run this command manually:")
            print(f"  {shlex.join(fetcher_command_args)}")
        print("✅ Successfully launched move fetcher process.")
    except Exception as e:
        print(f"\n--- ❌ An error occurred launching move_fetcher.py: {e} ---")
        print("Please run the command manually:")
        print(f"  {shlex.join(fetcher_command_args)}")

class WindowFinder:
    """Helper class to find a window handle (HWND) by its partial title."""
    def __init__(self, partial_title: str):
        self._hwnd = 0
        self.partial_title = partial_title.lower()

    def callback(self, hwnd, extra):
        """Callback function for win32gui.EnumWindows."""
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd).lower()
            if self.partial_title in window_title:
                self._hwnd = hwnd
    @property
    def hwnd(self):
        """Returns the found window handle."""
        return self._hwnd

def switch_to_window_robust(partial_title: str) -> bool:
    """Finds and activates a window using pywin32, bypassing focus-stealing prevention."""
    print(f"\n[ROBUST SWITCH] Searching for window: '{partial_title}'...")
    try:
        finder = WindowFinder(partial_title)
        win32gui.EnumWindows(finder.callback, None)
        hwnd = finder.hwnd
        if hwnd == 0:
            print(f"!!! ERROR: Window not found with title containing '{partial_title}'.")
            return False

        window_title = win32gui.GetWindowText(hwnd)
        print(f"Found window: '{window_title}' (HWND: {hwnd})")

        fg_hwnd = win32gui.GetForegroundWindow()
        fg_tid, _ = win32process.GetWindowThreadProcessId(fg_hwnd)
        current_tid = win32api.GetCurrentThreadId()
        win32process.AttachThreadInput(current_tid, fg_tid, True)

        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)

        win32process.AttachThreadInput(current_tid, fg_tid, False)
        time.sleep(0.1)

        if win32gui.GetForegroundWindow() == hwnd:
            print(f"SUCCESS: Window '{window_title}' is now active.")
            return True
        else:
            print(f"!!! WARNING: Activation failed for '{window_title}'.")
            return False
    except Exception as e:
        print(f"!!! CRITICAL ERROR in robust switch: {e}")
        return False


class GameController:
    """Manages the overall game state, engine communication, and GUI interaction."""
    # --- MODIFIED ---
    def __init__(self, window_config, colors: str, ponder: bool, use_tablebase: bool, tc_config: tuple, asymmetric_eval: bool, url: str):
        self.window_config = window_config
        self.ponder_enabled = ponder
        self.tablebase_enabled = use_tablebase
        self.asymmetric_eval = asymmetric_eval
        self.game_url = url

        self.tc_mode, self.tc_params = tc_config
        if self.tc_mode == 'dynamic':
            base_min, incr_sec = self.tc_params
            self.base_time_ms = base_min * 60 * 1000
            self.incr_time_ms = incr_sec * 1000
        else:
            self.fixed_time_ms = self.tc_params

        self.colors_arg = colors.lower()
        if len(self.colors_arg) != 2 or not all(c in 'rybg' for c in self.colors_arg):
             print(f"!!! CRITICAL: Invalid --colors argument '{colors}'. Must be two characters from 'r', 'y', 'b', 'g'.")
             sys.exit(1)

        self.controlled_colors = [c.upper() for c in self.colors_arg]
        self.self_partner_mode = (self.colors_arg[0] == self.colors_arg[1])

        if self.self_partner_mode:
            if self.colors_arg == 'rr': self.controlled_colors = ['R', 'Y']
            elif self.colors_arg == 'bb': self.controlled_colors = ['B', 'G']
            print(f"[CONFIG] Self-Partner mode enabled. Playing {self.controlled_colors} from a single window.")
        else:
            print(f"[CONFIG] Playing as colors: {self.controlled_colors[0]} and {self.controlled_colors[1]}.")

        self.windows = {}
        self._initialize_windows()

        self.board_configs = {}
        if not self.run_all_calibrations():
            print("CRITICAL: Board calibration failed. Exiting.")
            sys.exit(1)

        self.lock = threading.Lock()
        self.uci = uci_wrapper.UciWrapper(num_threads=9, max_depth=100, ponder=self.ponder_enabled)
        
        self.eval_log_file = EVAL_LOG_FILE
        self.current_move_evals = [".." for _ in range(4)]
        with open(self.eval_log_file, 'w') as f:
            f.write("--- 4PC Eval Log ---\n")
            f.write(f"Playing as: {self.controlled_colors}\n\n")
        
        self.reset_game()

        self.last_sync_time = 0
        self.stop_polling_event = threading.Event()
        self.polling_thread = threading.Thread(target=self._poll_game_state_file, daemon=True)
        self.polling_thread.start()

        self.game_is_over = False

        if 'R' in self.controlled_colors:
            print("[INIT] Engine is controlling Red. Proactively checking for first move...")
            initial_move_thread = threading.Thread(target=self._initial_move_thread_target, daemon=True)
            initial_move_thread.start()

    def shutdown(self):
        """Gracefully shuts down the controller and engine, saving any pending evals."""
        print("\nShutting down controller...")
        self.stop_polling_event.set()

        if self.game_is_over:
            self.uci.shutdown()
            return
        
        ply = len(self.board_moves)
        turn_index = ply % 4
        if turn_index > 0:
            move_number = (ply // 4) + 1
            move_num_str = f"{move_number}."
            logged_evals = self.current_move_evals[:turn_index]
            all_parts = [move_num_str] + logged_evals
            line = " ".join(all_parts) + "\n"
            with open(self.eval_log_file, 'a') as f:
                f.write(line)
                f.write("--- Shutdown mid-move ---\n")
        
        self.uci.shutdown()

    # --- MODIFIED ---
    def _log_final_evals_and_terminate_play(self):
        """
        Called when a game-ending position is detected. Stops engine activity,
        logs evaluations, says "gg" in the Bot-1 window, and signals shutdown.
        """
        print("[GAME END] Stopping engine and logging final evaluations...")
        self.uci.stop()

        ply = len(self.board_moves)
        turn_index = ply % 4
        
        if turn_index > 0:
            move_number = (ply // 4) + 1
            move_num_str = f"{move_number}."
            logged_evals = self.current_move_evals[:turn_index]
            all_parts = [move_num_str] + logged_evals
            line = " ".join(all_parts) + "\n"
            try:
                with open(self.eval_log_file, 'a') as f:
                    f.write(line)
                    f.write("--- Game Over ---\n")
                print(f"[EVAL LOG] Wrote final partial move evaluations to '{self.eval_log_file}'.")
            except IOError as e:
                print(f"!!! CRITICAL ERROR: Could not write final eval log: {e}")
        
        # --- SIMPLIFIED CHAT LOGIC ---
        print("\n[CHAT] Game is over. Attempting to say 'gg' in Bot-1 window.")
        bot1_window_title = self.window_config['BOT_1_TITLE']
        bot1_window = self._find_window(bot1_window_title)

        if bot1_window:
            time.sleep(0.05) # Give UI a moment to show the game over popup
            self._send_chat_message("gg", bot1_window)
        else:
            print(f"!!! WARNING: Bot-1 window ('{bot1_window_title}') not found. Cannot send 'gg'.")
        # ----------------------------
                
        self.current_move_evals = [".." for _ in range(4)]
        self.game_is_over = True
    
    def _find_chat_box(self, window: gw.Win32Window) -> Optional[Tuple[int, int]]:
        """Uses OCR to find the chat input box by its placeholder text."""
        print(f"[OCR] Searching for chat box in window: '{window.title}'...")
        try:
            screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
            ocr_data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)
            target_sequence = ['please', 'be', 'nice']
            
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes - len(target_sequence)):
                sequence_found = True
                for j in range(len(target_sequence)):
                    if ocr_data['text'][i+j].lower() != target_sequence[j]:
                        sequence_found = False
                        break
                
                if sequence_found:
                    (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                    center_x = x + w // 2
                    center_y = y + h // 2
                    abs_x = window.left + center_x
                    abs_y = window.top + center_y
                    print(f"[OCR] Found chat placeholder text at screen coordinates: ({abs_x}, {abs_y})")
                    return (abs_x, abs_y)

            print("[OCR] Could not find the chat box placeholder text.")
            return None
        except Exception as e:
            print(f"!!! CRITICAL OCR ERROR: {e}")
            print("    Please ensure Tesseract is installed and configured correctly.")
            return None

    # --- MODIFIED ---
    def _send_chat_message(self, message: str, window: gw.Win32Window):
        """Activates a window, finds the chat box, dismisses popups, and sends a message."""
        print(f"--- Sending chat message '{message}' to window '{window.title}' ---")
        activated = False
        
        for i in range(3): # Attempt up to 3 times
            if switch_to_window_robust(window.title):
                activated = True
                break
            print(f"!!! WARNING: Failed to activate window for chat (Attempt {i+1}/3). Retrying in 0.5s...")
            time.sleep(0.5)
        
        if not activated:
            print(f"!!! CRITICAL: Failed to activate window for chat after 3 attempts. Aborting message.")
            return
        
        time.sleep(0.05)

        print("[ACTION] Attempting to dismiss any popups by pressing ESC...")
        pyautogui.press('escape')
        time.sleep(0.05)
        
        # Find chat box coordinates just-in-time
        chat_coords = self._find_chat_box(window)
        if not chat_coords:
            print("!!! FAILED: Could not find chat box to send message. Aborting.")
            return

        try:
            pyautogui.click(chat_coords)
            pyautogui.click(chat_coords) # Twice to ensure focus
            time.sleep(0.05)
            pyautogui.write(message)
            time.sleep(0.05)
            pyautogui.press('enter')
            print(f"✅ Successfully sent chat message.")
        except Exception as e:
            print(f"!!! ERROR: Failed during pyautogui actions for chat: {e}")

    def _log_evaluation(self, ply: int, score: Optional[int]):
        """
        Logs the engine evaluation for a given ply. Now uses a simple,
        space-separated format without any fixed padding.
        """
        turn_index = ply % 4
        current_turn_char = ['R', 'B', 'Y', 'G'][turn_index]
        is_bg_turn = current_turn_char in ('B', 'G')

        if score is None:
            eval_str = ".."
        else:
            if abs(score) > 90000:
                if (score > 0 and not is_bg_turn) or (score < 0 and is_bg_turn):
                    eval_str = "+MATE"
                else:
                    eval_str = "-MATE"
            else:
                if is_bg_turn:
                    score *= -1
                score_pawns = score / 100.0
                eval_str = f"{score_pawns:+.1f}"

        self.current_move_evals[turn_index] = eval_str

        if turn_index == 3:
            move_number = (ply // 4) + 1
            move_num_str = f"{move_number}."
            
            all_parts = [move_num_str] + self.current_move_evals
            line = " ".join(all_parts) + "\n"

            try:
                with open(self.eval_log_file, 'a') as f:
                    f.write(line)
            except IOError as e:
                print(f"!!! CRITICAL ERROR: Could not write to eval log file: {e}")

            self.current_move_evals = [".." for _ in range(4)]

    def _poll_game_state_file(self):
        """Continuously polls the game state file for updates."""
        print("[POLLER] Starting file polling thread.")
        while not self.stop_polling_event.is_set():
            try:
                if os.path.exists(GAME_STATE_FILE):
                    mod_time = os.path.getmtime(GAME_STATE_FILE)
                    if mod_time > self.last_sync_time:
                        time.sleep(0.05)
                        with open(GAME_STATE_FILE, 'r') as f:
                            data = json.load(f)
                        
                        if data.get('url') != self.game_url:
                            continue

                        if data.get('detection_timestamp', 0) > self.last_sync_time:
                            self.last_sync_time = data['detection_timestamp']
                            self.sync_board_state(data['moves'], data.get('clocks', {}))
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"[POLLER] Warning: Could not read game state file ({e}). Retrying.")
            except Exception as e:
                print(f"[POLLER] CRITICAL ERROR in polling thread: {e}")

            time.sleep(POLL_INTERVAL_SECONDS)
        print("[POLLER] Polling thread stopped.")

    def _initialize_windows(self):
        """Initializes window handles based on --colors argument."""
        bot1_title = self.window_config['BOT_1_TITLE']
        bot2_title = self.window_config['BOT_2_TITLE']

        if self.self_partner_mode:
            bot1_window = self._find_window(bot1_title)
            self.windows[self.controlled_colors[0]] = bot1_window
            self.windows[self.controlled_colors[1]] = bot1_window
            print(f"Mapping colors {self.controlled_colors[0]} and {self.controlled_colors[1]} to window '{bot1_title}'")
        else:
            self.windows[self.controlled_colors[0]] = self._find_window(bot1_title)
            self.windows[self.controlled_colors[1]] = self._find_window(bot2_title)
            print(f"Mapping color {self.controlled_colors[0]} to window '{bot1_title}'")
            print(f"Mapping color {self.controlled_colors[1]} to window '{bot2_title}'")

        if not all(self.windows.values()):
            print("!!! WARNING: Could not find one or more required game windows at startup.")

    def run_all_calibrations(self) -> bool:
        """Calibrates all unique perspectives required by the controlled colors."""
        perspectives_to_calibrate: Set[str] = set()
        for color in self.controlled_colors:
            if color in ('R', 'Y'): perspectives_to_calibrate.add('R')
            elif color in ('B', 'G'): perspectives_to_calibrate.add('B')

        print(f"\n--- Required Calibrations: {list(perspectives_to_calibrate)} ---")

        for perspective_key in perspectives_to_calibrate:
            calib_color = perspective_key if perspective_key in self.controlled_colors else ('Y' if perspective_key == 'R' else 'G')
            target_window = self.windows.get(calib_color)

            if not target_window:
                print(f"!!! CRITICAL: Cannot calibrate perspective '{perspective_key}' because its window was not found.")
                return False

            print(f"\n--- Starting Calibration for Perspective: '{perspective_key}' (using window '{target_window.title}') ---")
            if not switch_to_window_robust(target_window.title):
                 print(f"!!! CRITICAL: Failed to activate window '{target_window.title}' for calibration.")
                 return False

            locator = BoardLocator(piece_png_path=PIECE_ASSET_PATH, window_title=target_window.title)
            geometry = locator.calibrate()
            if geometry:
                self.board_configs[perspective_key] = geometry
                print(f"--- Calibration for '{perspective_key}' Successful ---")
            else:
                print(f"--- Calibration for '{perspective_key}' FAILED ---")
                return False
        return True

    def _initial_move_thread_target(self):
        """Waits briefly then checks for the first move, in case Red is controlled."""
        time.sleep(3)
        with self.lock:
            if not self.board_moves:
                print("[THREAD] No moves detected on startup, initiating first move check...")
                self.check_and_play()
            else:
                print("[THREAD] Moves were already synced, initial check is not needed.")

    def reset_game(self):
        """Resets the internal game state to the beginning."""
        with self.lock:
            print("\n--- GAME RESET ---")
            self.uci.set_position(uci_wrapper.START_FEN_NEW)
            self.board_moves = []
            self.clock_times_sec = {}
            self.current_move_evals = [".." for _ in range(4)]
            print(f"Internal board state has been reset. Playing as colors: {self.colors_arg}")

    def _log_clocks(self):
        """Prints the current clock times for all players in a formatted string."""
        if not self.clock_times_sec:
            return
        clock_strings = []
        for color in ['R', 'B', 'Y', 'G']:
            time_sec = self.clock_times_sec.get(color, 0)
            minutes, seconds = divmod(int(time_sec), 60)
            clock_strings.append(f"{color}: {minutes:01d}:{seconds:02d}")
        print(f"[CLOCKS] " + " | ".join(clock_strings))

    def sync_board_state(self, incoming_moves: List[str], incoming_clocks: dict):
        """Receives new game state, updates internals, and triggers a move check."""
        with self.lock:
            if incoming_clocks: self.clock_times_sec = incoming_clocks
            if incoming_moves != self.board_moves:
                print(f"\n[SYNC] Received {len(incoming_moves)} moves, had {len(self.board_moves)}. Updating state.")
                self.board_moves = incoming_moves
                self._log_clocks()

                self.uci.set_position(self.get_current_fen(), self.board_moves)
                num_legal_moves = self.uci.get_num_legal_moves()

                if num_legal_moves is not None and num_legal_moves == 0:
                    print("\n--- GAME OVER DETECTED (0 legal moves available) ---")
                    self._log_final_evals_and_terminate_play()
                    return

                self.check_and_play()

    def _calculate_dynamic_time_ms(self, current_turn_char: str) -> int:
        """Calculates the optimal time to think for a move based on clock and increment."""
        if len(self.board_moves) < 4:
            print("[TIME] Opening move (<4), using fixed 5s.")
            return 5000
        clock_sec = self.clock_times_sec.get(current_turn_char)
        if clock_sec is None:
            print(f"[TIME] No clock data for {current_turn_char}, using default 5s.")
            return 5000
        clock_ms = clock_sec * 1000
        move_time_ms = self.incr_time_ms + (clock_ms - MIN_REMAINING_MOVE_MS) / TIME_DIVISOR if clock_ms > MIN_REMAINING_MOVE_MS else self.incr_time_ms
        move_time_ms = int(min(max(move_time_ms, MIN_MOVE_TIME_MS), MAX_MOVE_MS) * SAFETY_MARGIN)
        print(f"[TIME] Clock: {clock_sec:.1f}s. Calculated think time: {move_time_ms / 1000:.2f}s.")
        return move_time_ms

    def get_time_to_think_ms(self, current_turn_char: str) -> int:
        """Determines think time based on the configured time control mode."""
        if self.tc_mode == 'fixed':
            print(f"[TIME] Using fixed time control: {self.fixed_time_ms}ms.")
            return self.fixed_time_ms
        return self._calculate_dynamic_time_ms(current_turn_char)

    def check_and_play(self):
        """
        The main decision-making loop. Logs a placeholder for every opponent
        turn to ensure the log file is always written correctly.
        """
        turn_order = ['R', 'B', 'Y', 'G']
        ply_number = len(self.board_moves)
        current_turn_char = turn_order[ply_number % 4]

        if current_turn_char in self.controlled_colors:
            print(f"--- My turn to play as {current_turn_char} (controlled colors: {self.colors_arg}) ---")
            
            if self.ponder_enabled:
                self.uci.stop()
            
            perspective = 'R' if self.self_partner_mode and 'R' in self.controlled_colors else \
                        'B' if self.self_partner_mode and 'B' in self.controlled_colors else \
                        current_turn_char

            if self.tablebase_enabled and (best_move := tablebase.get_tablebase_move(self.board_moves)):
                print(f"Found move in tablebase: {best_move}")
                self._log_evaluation(ply_number, None)
                self.execute_gui_move(best_move, perspective=perspective)
                return

            time_to_think_ms = self.get_time_to_think_ms(current_turn_char)
            if self.asymmetric_eval:
                self.uci.set_team('red_yellow' if current_turn_char in 'RY' else 'blue_green')

            self.uci.set_position(self.get_current_fen(), self.board_moves)
            result = self.uci.get_best_move(time_limit_ms=time_to_think_ms, gameover_callback=lambda: None)

            if 'best_move' in result:
                print(f"Engine chose move: {result['best_move']}")
                score = result.get('score') 
                self._log_evaluation(ply_number, score)
                self.execute_gui_move(result['best_move'], perspective=perspective)
            else:
                print("!!! WARNING: Engine did not return a move.")
                self._log_evaluation(ply_number, None)
        else:
            self._log_evaluation(ply_number, None)
            
            if self.ponder_enabled:
                self.uci.stop() 
                if self.asymmetric_eval:
                    self.uci.set_team('red_yellow' if current_turn_char in 'RY' else 'blue_green')
                self.uci.set_position(self.get_current_fen(), self.board_moves)
                print(f"--- Opponent's turn ({current_turn_char}). Starting to ponder... ---")
                self.uci.start_pondering(gameover_callback=lambda: None)

    def execute_gui_move(self, move_str: str, perspective: str):
        """Converts an algebraic move to pixel coordinates and executes it via GUI automation."""
        playing_color = ['R', 'B', 'Y', 'G'][len(self.board_moves) % 4]
        target_window = self.windows.get(playing_color)
        if not target_window or not switch_to_window_robust(target_window.title):
            print(f"!!! CRITICAL: Failed to activate window for {playing_color}. Aborting move.")
            return

        from_pos, to_pos = move_str.split('-')[0], move_str.split('-')[1].split('=')[0]
        from_px = self.algebraic_to_pixels(from_pos, perspective)
        to_px = self.algebraic_to_pixels(to_pos, perspective)

        print(f"Clicking from {from_pos} at {from_px} to {to_pos} at {to_px} (Visual Perspective: {perspective})")
        pyautogui.click(from_px)
        time.sleep(0.025)
        pyautogui.dragTo(to_px[0], to_px[1], duration=0.05, button='left')

    def get_current_fen(self) -> str:
        """Returns the FEN for the current game position."""
        return uci_wrapper.START_FEN_NEW

    def _find_window(self, title: str) -> Optional[gw.Win32Window]:
        """Finds a window by its partial title using pygetwindow."""
        try:
            wins = [w for w in gw.getAllWindows() if title.lower() in w.title.lower()]
            if not wins:
                print(f"Warning: Window with title '{title}' not found.")
            return wins[0] if wins else None
        except Exception as e:
            print(f"Error finding window: {e}")
            return None

    def algebraic_to_pixels(self, algebraic_pos: str, perspective: str) -> Tuple[int, int]:
        """Converts an algebraic board position (e.g., 'a1') to screen pixel coordinates."""
        base_perspective = 'R' if perspective in ['R', 'Y'] else 'B'
        board_config = self.board_configs.get(base_perspective)
        if not board_config:
            print(f"!!! CRITICAL ERROR: No board config for base perspective '{base_perspective}'.")
            return (0, 0)
        col, row = ord(algebraic_pos[0]) - ord('a'), int(algebraic_pos[1:]) - 1
        if perspective in ('Y', 'G'):
            col, row = (BOARD_DIMENSIONS - 1) - col, (BOARD_DIMENSIONS - 1) - row
        px = board_config['origin_a1_center_px'] + col * board_config['unit_vec_col'] + row * board_config['unit_vec_row']
        return tuple(px.astype(int))

def _parse_tc(value: str) -> tuple[str, Union[tuple[float, float], int]]:
    """
    Parses time control strings for argparse.
    - 'minutes+increment' (e.g., '2+10') -> ('dynamic', (2.0, 10.0))
    - 'fixed_ms' (e.g., 'fixed_3000') -> ('fixed', 3000)
    """
    if value.startswith('fixed_'):
        try:
            return ('fixed', int(value.split('_')[1]))
        except (ValueError, IndexError):
            raise argparse.ArgumentTypeError("Fixed TC must be in 'fixed_milliseconds' format (e.g., 'fixed_3000').")
    elif '+' in value:
        try:
            p = value.split('+')
            if len(p) != 2: raise ValueError()
            return ('dynamic', (float(p[0]), float(p[1])))
        except ValueError:
            raise argparse.ArgumentTypeError("Dynamic TC must be in 'minutes+increment' format (e.g., '2+10').")
    raise argparse.ArgumentTypeError("Invalid TC format. Use 'M+S' or 'fixed_ms'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="4-Player Chess Game Controller Bot. This script is the main entry point.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--colors', type=str, required=True,
        help="Specify which two colors the engine will play for. \n"
             "The first color uses the 'Bot-1' window, the second uses the 'Bot-2' window.\n"
             "Examples:\n"
             " 'ry' -> Red(Bot-1) & Yellow(Bot-2) team\n"
             " 'bg' -> Blue(Bot-1) & Green(Bot-2) team\n"
             " 'rb' -> Mixed team: Red(Bot-1) & Blue(Bot-2)\n"
             " 'rr' -> Self-Partner: Red & Yellow play from Bot-1's window\n"
             " 'bb' -> Self-Partner: Blue & Green play from Bot-1's window"
    )
    parser.add_argument('--url', required=True, help="The full URL of the chess.com game to fetch moves and clock times from.")
    parser.add_argument(
        '--tc', type=_parse_tc, default='2+10',
        help="Time control. Two formats are supported:\n"
             " - Dynamic: 'minutes+increment' (e.g., '2+10', '1.5+5')\n"
             " - Fixed:   'fixed_milliseconds' (e.g., 'fixed_3000')\n"
             "Default is '2+10'."
    )
    parser.add_argument('--asymmetric_eval', action='store_true', help="Enable asymmetric evaluation.")
    parser.add_argument('--ponder', action='store_true', help="Enable thinking on the opponent's turn.")
    parser.add_argument('--tablebase', dest='tablebase', action='store_true', help="Enable opening tablebase.")
    parser.add_argument('--no-tablebase', dest='tablebase', action='store_false', help="Disable opening tablebase.")
    parser.set_defaults(tablebase=True)

    args = parser.parse_args()

    print(f"Initializing clean game state file for URL: {args.url}")
    initial_payload = {
        'url': args.url,
        'moves': [],
        'clocks': {},
        'detection_timestamp': time.time()
    }
    try:
        temp_file = GAME_STATE_FILE + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(initial_payload, f)
        os.replace(temp_file, GAME_STATE_FILE)
        print(f"'{GAME_STATE_FILE}' has been reset.")
    except Exception as e:
        print(f"!!! CRITICAL: Could not initialize game state file: {e}")
        sys.exit(1)

    launch_move_fetcher(args.url)
    print("Waiting a moment for the fetcher to launch...")
    time.sleep(2)

    controller = GameController(
        WINDOW_CONFIG, colors=args.colors, ponder=args.ponder,
        use_tablebase=args.tablebase, tc_config=args.tc,
        asymmetric_eval=args.asymmetric_eval, url=args.url
    )

    tc_mode, tc_params = args.tc
    if tc_mode == 'dynamic':
        print(f"\n[CONFIG] Time control set to DYNAMIC: {tc_params[0]} minutes + {tc_params[1]} seconds per move.")
    else:
        print(f"\n[CONFIG] Time control set to FIXED: {tc_params}ms per move.")

    if args.asymmetric_eval: print("[CONFIG] Asymmetric evaluation has been ENABLED.")
    else: print("[CONFIG] Asymmetric evaluation is DISABLED (standard evaluation).")
    if args.ponder: print("[CONFIG] Pondering is ENABLED.")
    else: print("[CONFIG] Pondering is DISABLED.")
    if args.tablebase: print("[CONFIG] Opening move tablebase is ENABLED.")
    else: print("[CONFIG] Opening move tablebase is DISABLED.")


    print("\n--- 4-Player Chess Game Controller ---")
    print(f"Running for colors: {args.colors.upper()}")
    print("A separate terminal has been launched for the move fetcher.")
    print("Controller is running. Press Ctrl+C to exit.")
    
    try:
        while not controller.game_is_over:
            time.sleep(1)
        print("\nGame has ended. Shutting down gracefully.")
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()