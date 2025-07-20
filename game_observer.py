# game_observer.py
# v1.2 (Adds clean shutdown method)

import sys
import time
import threading
import argparse
from typing import List
import json
import os

import subprocess
import platform
import shlex

import uci_wrapper

GAME_STATE_FILE = "game_state.json"
POLL_INTERVAL_SECONDS = 0.2 # Observer can poll a bit slower

def launch_move_fetcher(url: str):
    """Launches move_fetcher.py in a separate terminal based on the OS."""
    current_os = platform.system()
    print(f"[{current_os.upper()}] Launching move fetcher in a new terminal...")
    fetcher_command_args = [sys.executable, "move_fetcher.py", "--url", url]
    try:
        if current_os == "Windows":
            subprocess.Popen(['start', 'cmd', '/k'] + fetcher_command_args, shell=True)
        elif current_os == "Darwin":
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


class GameObserver:
    def __init__(self, url: str):
        self.lock = threading.Lock()
        self.game_url = url
        self.uci = uci_wrapper.UciWrapper(num_threads=8, max_depth=100, ponder=True)
        self.reset_game()
        
        self.last_sync_time = 0
        self.stop_polling_event = threading.Event()
        self.polling_thread = threading.Thread(target=self._poll_game_state_file, daemon=True)
        self.polling_thread.start()

    # **NEW METHOD**
    def shutdown(self):
        print("\nShutting down observer...")
        self.stop_polling_event.set()
        self.uci.shutdown()

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
                            self.sync_board_state(data['moves'])
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                pass # Ignore errors, just retry
            except Exception as e:
                print(f"[POLLER] CRITICAL ERROR in polling thread: {e}")

            time.sleep(POLL_INTERVAL_SECONDS)
        print("[POLLER] Polling thread stopped.")

    def reset_game(self):
        with self.lock:
            print("--- OBSERVER RESET ---")
            self.uci.set_position(uci_wrapper.START_FEN_NEW)
            self.board_moves = []
            print("Internal board state has been reset.")

    def sync_board_state(self, incoming_moves: List[str]):
        with self.lock:
            if incoming_moves != self.board_moves:
                print(f"\n[SYNC] Received new position with {len(incoming_moves)} moves. Analyzing...")
                self.board_moves = incoming_moves
                self.analyze_position()

    def analyze_position(self):
        """
        The core logic for the observer. It stops any previous analysis,
        sets the new position, and starts a new infinite search.
        """
        self.uci.stop()
        self.uci.set_position(uci_wrapper.START_FEN_NEW, self.board_moves)
        
        print("--- Engine is analyzing (pondering)... ---")
        self.uci.start_pondering(gameover_callback=lambda: print("\n--- Game Over Detected ---"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="4-Player Chess Game Observer. Watches a game and shows engine evaluation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--url',
        required=True,
        help="The full URL of the chess.com game to observe."
    )
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

    observer = GameObserver(url=args.url)

    print("\n--- 4-Player Chess Game Observer ---")
    print("A separate terminal has been launched for the move fetcher.")
    print("Engine output (evaluation and PV) will be shown below as moves are made.")
    print("Press Ctrl+C to exit.")
    
    observer.analyze_position()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # **UPDATED SHUTDOWN LOGIC**
        observer.shutdown()