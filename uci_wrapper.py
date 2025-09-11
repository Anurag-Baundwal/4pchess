# uci_wrapper.py
# v0.4 (Implements non-blocking queue-based reader for responsive stop commands)

import time
import os
import subprocess
import threading
import re
from typing import Callable, Optional, List
import queue # Import the queue library

# FENs for starting positions (content unchanged)
START_FEN_RBG = "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x"
START_FEN_NEW = "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x"
START_FEN_BYG = "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yQ,yK,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x"
START_FEN_OLD = "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bK,bP,10,gP,gQ/bQ,bP,10,gP,gK/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x"
START_FEN_BY = "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yQ,yK,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gQ/bK,bP,10,gP,gK/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x"


class UciWrapper:
    def __init__(self, num_threads, max_depth, ponder):
        self._num_threads = num_threads
        self._max_depth = max_depth
        self._ponder = ponder
        self._process: Optional[subprocess.Popen] = None
        self._team = None
        self._lock = threading.Lock() # Lock for sending commands to the engine (stdin)

        # --- NEW: Queue for receiving messages from the engine ---
        self._output_queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_reader_event = threading.Event()
        # --------------------------------------------------------

        self.is_searching = False # State to track if a 'go' command is active

        self.create_process()

    def _enqueue_output(self, process):
        """
        This function runs in a separate thread. Its only job is to
        block on `readline` and put the received line into the queue.
        """
        for line in iter(process.stdout.readline, ''):
            if self._stop_reader_event.is_set():
                break
            self._output_queue.put(line.strip())
        print("[READER THREAD] Exiting.")

    def create_process(self):
        with self._lock:
            if self._process and self._process.poll() is None:
                try:
                    self._stop_reader_event.set()
                    self._process.terminate()
                    self._process.wait(timeout=1.0)
                    if self._reader_thread and self._reader_thread.is_alive():
                        self._reader_thread.join(timeout=1.0)
                except (subprocess.TimeoutExpired, Exception):
                    self._process.kill()
            
            print("Creating new engine process...")
            self._process = subprocess.Popen(
                os.path.join(os.getcwd(), 'cli'),
                universal_newlines=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1)
            
            # --- Start the new reader thread ---
            self._stop_reader_event.clear()
            self._reader_thread = threading.Thread(target=self._enqueue_output, args=(self._process,))
            self._reader_thread.daemon = True
            self._reader_thread.start()
            print("[READER THREAD] Started.")
            # ------------------------------------

            self._send_command_internal(f'setoption name threads value {self._num_threads}')

    def __del__(self):
        self.shutdown()

    def _send_command_internal(self, command: str):
        # Assumes lock is already held
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.write(command + '\n')
                self._process.stdin.flush()
            except (IOError, ValueError):
                pass
    
    def _send_command(self, command: str):
        with self._lock:
            self._send_command_internal(command)

    def maybe_recreate_process(self):
        # No lock needed here as create_process handles its own locking
        if self._process is None or self._process.poll() is not None:
            print('Engine process is dead or missing. Recreating...')
            self.create_process()
            if self._team:
                self.set_team(self._team)

    def stop(self):
        """Stops an active search. Now non-blocking and highly responsive."""
        if self.is_searching:
            print("[UCI] Sending 'stop' command to engine...")
            self._send_command('stop')
            # We don't wait here. The get_best_move loop will handle the 'bestmove' response.
            # self.is_searching will be set to False when 'bestmove' is received.
    
    def shutdown(self):
        """Gracefully shuts down the engine process."""
        print("Shutting down engine process...")
        self._stop_reader_event.set()
        with self._lock:
            if not self._process or self._process.poll() is not None:
                self._process = None
                return
            try:
                self._send_command_internal('quit')
                self._process.wait(timeout=1.0)
            except (subprocess.TimeoutExpired, IOError, ValueError):
                print("Engine did not quit gracefully, killing process.")
                self._process.kill()
            finally:
                self._process = None
        
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)


    def set_team(self, team):
        if team != self._team:
            print('set team:', team)
            self._team = team
            if self._team is not None:
                self._send_command(f'setoption name engine_team value {team}')

    def start_pondering(self, gameover_callback: Callable[[], None]):
        """Starts an infinite search. Can now be properly stopped."""
        if not self._ponder or self.is_searching:
            return

        self.maybe_recreate_process()
        self._send_command('go infinite')
        self.is_searching = True
        
        # We don't need a separate thread here anymore. The main polling loop
        # in the controller will continue, and the stop command will work correctly.
        print("[UCI] Pondering started. Send stop() to halt.")


    def set_position(self, fen: str, moves: Optional[List[str]] = None):
        self.maybe_recreate_process()
        fen = fen.replace('\n', '')
        parts = [f'position fen {fen}']
        if moves:
            parts.append('moves')
            parts.extend(moves)
        self._send_command(' '.join(parts))

    def get_num_legal_moves(self):
        self._send_command('get_num_legal_moves')
        while True:
            try:
                line = self._output_queue.get(timeout=1.0) # Wait up to 1s for a response
                if 'info string n_legal' in line:
                    m = re.search(r'n_legal (\d+)', line)
                    if m: return int(m.group(1))
            except queue.Empty:
                print("Warning: Timed out waiting for num_legal_moves response.")
                return None
        return None

    def get_best_move(
        self,
        time_limit_ms: int,
        gameover_callback: Callable[[], None],
        pv_callback: Optional[Callable[[list[str]], None]] = None,
        depth_callback: Optional[Callable[[int, int, str], None]] = None,
        last_move: Optional[str] = None
    ):
        self.maybe_recreate_process()
        if self.is_searching:
            self.stop() # Stop any previous search (like pondering)

        n_legal = self.get_num_legal_moves()
        max_depth = self._max_depth
        if n_legal == 1:
            max_depth = 1
            print('Forced move')

        msg = f'go movetime {time_limit_ms}'
        if max_depth is not None:
            msg += f' depth {max_depth}'
        
        # --- Clear the queue of any old messages before starting ---
        while not self._output_queue.empty():
            try: self._output_queue.get_nowait()
            except queue.Empty: break
        # -----------------------------------------------------------
        
        self._send_command(msg)
        self.is_searching = True
        response = {}

        # --- NEW NON-BLOCKING RESPONSE LOOP ---
        start_time = time.time()
        timeout_sec = (time_limit_ms / 1000.0) + 4.0 # Generous timeout

        while self.is_searching and (time.time() - start_time) < timeout_sec:
            try:
                # Wait for a short time, allowing this loop to be responsive to changes
                line = self._output_queue.get(timeout=0.1) 
                print(line)

                if 'Game completed' in line:
                    response['gameover'] = True
                    if gameover_callback: gameover_callback()

                if ' pv ' in line:
                    m = re.search(r'pv (.*?) score ([-\d]+)', line)
                    if m:
                        pv = m.group(1).split()
                        score = int(m.group(2))
                        current_move = pv[0]

                        response['best_move'] = current_move
                        response['pv'] = pv
                        response['score'] = score

                        if (d_match := re.search(r'depth (\d+)', line)):
                            depth = int(d_match.group(1))
                            response['depth'] = depth
                            if depth_callback:
                                depth_callback(depth, score, current_move)
                
                if line.startswith('bestmove'):
                    m = re.search('bestmove (.*)', line)
                    if m: response['best_move'] = m.group(1)
                    self.is_searching = False # The search is officially over
                    break # Exit the loop immediately

            except queue.Empty:
                # This is normal. It means no new message from the engine in the last 0.1s.
                # The loop continues, checking the timeout and is_searching conditions.
                continue
        # --- END OF NEW LOOP ---

        if self.is_searching:
            print('Warning: Search timed out. Recreating process.')
            self.create_process() # Recreate process if it becomes unresponsive
        self.is_searching = False

        return response