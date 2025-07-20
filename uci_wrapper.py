# uci_wrapper.py
# v0.3.1 (Improves shutdown logic to be silent and more robust)

import time
import os
import subprocess
import threading
import re
from typing import Callable, Optional, List

# FENs for starting positions (content unchanged)
START_FEN_RBG = "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rK,rQ,rB,rN,rR,x,x,x"
START_FEN_NEW = "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x"
START_FEN_BYG = "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yQ,yK,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x"
START_FEN_OLD = "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bK,bP,10,gP,gQ/bQ,bP,10,gP,gK/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x"
START_FEN_BY = "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yQ,yK,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gQ/bK,bP,10,gP,gK/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x"

def get_move_response(lock, process, response, pv_callback, gameover_callback):
  while process.poll() is None:
    try:
        with lock:
            line = process.stdout.readline().strip()
        
        if not line:
            break
        print(line)
        if 'Game completed' in line:
            response['gameover'] = True
            if gameover_callback: gameover_callback()
        if ' pv ' in line:
            m = re.search(r'pv (.*?) score ([-\d]+)', line)
            if m:
                pv = m.group(1).split()
                score = m.group(2)
                response['best_move'] = pv[0]
                response['pv'] = pv
                response['score'] = int(score)
            if (d_match := re.search(r'depth (\d+)', line)): response['depth'] = int(d_match.group(1))
            if (t_match := re.search(r'time (\d+)', line)):
                movetime = int(t_match.group(1))
                depth = response.get('depth', 0)
                if pv_callback and (depth >= 15 or (movetime >= 500 and depth >= 10)):
                    pv_callback(pv)
        if 'bestmove' in line:
            m = re.search('bestmove (.*)', line)
            if m: response['best_move'] = m.group(1)
            break
    except (IOError, ValueError):
        # This can happen if the process is terminated while reading
        break


class UciWrapper:
    def __init__(self, num_threads, max_depth, ponder):
        self._num_threads = num_threads
        self._max_depth = max_depth
        self._ponder = ponder
        self._process: Optional[subprocess.Popen] = None
        self._ponder_thread: Optional[threading.Thread] = None
        self._team = None
        self._lock = threading.Lock() # Lock for all subprocess communication
        self.create_process()

    def create_process(self):
        with self._lock:
            if self._process and self._process.poll() is None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=1.0)
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
                # This can happen if the process dies between the poll check and the write
                # It's not a critical error during shutdown, so we don't print anything here.
                pass
    
    def _send_command(self, command: str):
        with self._lock:
            self._send_command_internal(command)

    def maybe_recreate_process(self):
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                print('Engine process is dead or missing. Recreating...')
                self.create_process()
                if self._team:
                    self.set_team(self._team)

    def stop(self):
        """Stops an active ponder search and waits for it to finish."""
        if self._ponder_thread and self._ponder_thread.is_alive():
            print("Sending 'stop' command to engine...")
            self._send_command('stop')
            try:
                self._ponder_thread.join(timeout=2.0)
                if self._ponder_thread.is_alive():
                    print("!!! WARNING: Ponder thread did not join in time after 'stop'.")
                else:
                    print("Ponder display thread stopped.")
            except Exception as e:
                print(f"Error joining ponder thread: {e}")
            self._ponder_thread = None
    
    def shutdown(self):
        """Gracefully shuts down the engine process."""
        with self._lock:
            if not self._process or self._process.poll() is not None:
                # Process is already gone or never existed, nothing to do.
                self._process = None
                return

            print("Shutting down engine process...")
            try:
                self._send_command_internal('quit')
                # Wait for a graceful exit.
                self._process.wait(timeout=1.0)
            except (subprocess.TimeoutExpired, IOError, ValueError):
                # If graceful quit fails or times out, kill it.
                print("Engine did not quit gracefully, killing process.")
                self._process.kill()
            finally:
                # Ensure the process object is cleaned up.
                self._process = None

    def set_team(self, team):
        if team != self._team:
            print('set team:', team)
            self._team = team
            if self._team is not None:
                self._send_command(f'setoption name engine_team value {team}')

    def start_pondering(self, gameover_callback: Callable[[], None]):
        """Starts an infinite search on the currently set position."""
        if not self._ponder or (self._ponder_thread and self._ponder_thread.is_alive()):
            return

        self.maybe_recreate_process()
        self._send_command('go infinite')
        
        self._ponder_thread = threading.Thread(
            target=get_move_response,
            args=(self._lock, self._process, {}, lambda pv: None, gameover_callback)
        )
        print("Ponder display thread started.")
        self._ponder_thread.start()

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
            with self._lock:
                if self._process.poll() is not None: return None
                try:
                    line = self._process.stdout.readline().strip()
                except (IOError, ValueError):
                    return None
            if 'info string n_legal' in line:
                m = re.search(r'n_legal (\d+)', line)
                if m: return int(m.group(1))
        return None

    def get_best_move(
        self,
        time_limit_ms: int,
        gameover_callback: Callable[[], None],
        pv_callback: Optional[Callable[[list[str]], None]] = None,
        last_move: Optional[str] = None
    ):
        self.maybe_recreate_process()
        self.stop() 
        
        n_legal = self.get_num_legal_moves()
        max_depth = self._max_depth
        if n_legal == 1:
            max_depth = 1
            print('Forced move')

        pre_search_start_time = time.time()
        buffer_ms = 50
        min_move_ms = 10
        time_limit_ms = max(time_limit_ms - buffer_ms, min_move_ms)

        msg = f'go movetime {time_limit_ms}'
        if max_depth is not None:
            msg += f' depth {max_depth}'
        
        print(f"[TIMER]   ┗> UCI: {(time.time() - pre_search_start_time):.3f}s before sending 'go' command.")
        self._send_command(msg)

        response = {}
        reader_thread = threading.Thread(
            target=get_move_response,
            args=(self._lock, self._process, response, pv_callback, gameover_callback))
        reader_thread.start()
        
        timeout_sec = (time_limit_ms / 1000.0) + 2.0
        reader_thread.join(timeout_sec)

        if reader_thread.is_alive():
            print('Warning: UCI subprocess timed out during timed search. Recreating process.')
            self.create_process()

        if not response.get('gameover') and 'best_move' not in response:
            print(f"Warning: Best move not found in response: {response}. This can happen after recovery.")

        print(f"[TIMER]   ┗> UCI: {(time.time() - pre_search_start_time):.3f}s total for get_best_move wrapper.")
        return response