"""Server for playing on chess.com."""

import api
import argparse
import json
import re
import requests
import tablebase
import time
import uci_wrapper_old as uci_wrapper

parser = argparse.ArgumentParser(
    prog='Server',
    description='Interacts with chess.com')
def parse_bool(x):
  return x.lower() in ['true', 't', '1']
parser.add_argument('-prod', '--prod', type=parse_bool, required=False,
    default=True)
# might need to set this to another value depending on your computer.
# recommended: max(1, #physical processors - 2)
parser.add_argument('-num_threads', '--num_threads', type=int, required=False,
    default=9)
parser.add_argument('-max_depth', '--max_depth', type=int, required=False,
    default=100)
parser.add_argument('-arrows', '--arrows', type=parse_bool, required=False,
    default=False)
# Enables asymmetric evaluation (anti-human eval)
parser.add_argument(
    '-asymmetric_eval', '--asymmetric_eval', type=parse_bool, required=False,
    default=False)
parser.add_argument(
    '-chat_eval', '--chat_eval', type=parse_bool, required=False,
    default=False)
parser.add_argument(
    '-enable_tablebase', '--enable_tablebase', type=parse_bool, required=False,
    default=True)
parser.add_argument(
    '-ponder', '--ponder', type=parse_bool, required=False,
    default=False)
parser.add_argument(
    '-play_fast', '--play_fast', type=parse_bool, required=False,
    default=False)

# --- NNUE specific arguments ---
parser.add_argument(
    '-enable_nnue', '--enable_nnue', type=parse_bool, required=False,
    default=False)
parser.add_argument(
    '-nnue_path', '--nnue_path', type=str, required=False,
    default='nnue/models')
# -------------------------------
args = parser.parse_args()


_USE_PROD_SERVER = args.prod
_BOT_NAME = 'TeamEnigma'
_BOT_VERSION = 'v0.0.0'
if _USE_PROD_SERVER:
  _API_KEY_FILENAME = 'api_key_prod.txt'
  _SERVER_URL = api.MAIN_SERVER_URL
else:
  _API_KEY_FILENAME = 'api_key_test.txt'
  _SERVER_URL = api.TEST_SERVER_URL

_MAX_MOVE_MS = 30000
_MIN_REMAINING_MOVE_MS = 0
_MIN_MOVE_TIME_MS = 100

# --- TIME MANAGEMENT CONSTANTS ---
_TIME_PRESSURE_THRESHOLD_PERCENT = 0.50
_TIME_PRESSURE_FACTOR_DECREASE = 0.75
_SAFETY_MARGIN = 0.95
# ------------------------------------

def _read_api_token(filepath: str) -> str:
  """Read the API token from the given filepath."""
  with open(filepath) as f:
    return f.read().strip()


def _standardize_move(move: str) -> str:
  move = ''.join([x for x in move if x != '+'])
  separator = '-'
  if 'x' in move:
    separator = 'x'
  parts = move.split(separator)
  for i in range(len(parts)):
    part = parts[i]
    m = re.match('^([a-zA-Z]+)', part)
    if m:
      letters = m.group(1)
      if len(letters) > 1:
        parts[i] = part[len(letters) - 1:]
  return '-'.join(parts)


class Pgn4Info:

  def __init__(self, base_time_ms: int, incr_time_ms: int, delay_time_ms: int,
      last_move: str | None, team: str, played_n_moves: int,
      game_number: str, my_colors: list[str]):
    self.base_time_ms = base_time_ms
    self.incr_time_ms = incr_time_ms
    self.delay_time_ms = delay_time_ms
    self.last_move = last_move
    self.team = team
    self.played_n_moves = played_n_moves
    self.game_number = game_number
    self.my_colors = my_colors

  @classmethod
  def FromString(cls, pgn4: str):
    m = re.search('TimeControl "(.*?)\\+(.*?)"', pgn4)
    if not m:
      return None
    base_time_mins = float(m.group(1))
    extra_time_secs = m.group(2)
    delay_time_secs = 0
    incr_time_secs = 0
    if extra_time_secs.endswith('D'):
      delay_time_secs = float(extra_time_secs[:-1])
    else:
      incr_time_secs = float(extra_time_secs)

    base_time_ms = int(base_time_mins * 60 * 1000)
    incr_time_ms = int(incr_time_secs * 1000)
    delay_time_ms = int(delay_time_secs * 1000)

    pattern = 'GameNr "(.*?)"'
    game_number = None
    m = re.search(pattern, pgn4, re.DOTALL)
    if m:
      game_number = m.group(1)

    last_move = None
    lines = pgn4.split('\n')
    last_line = lines[-1]
    pattern = r'^(\d+)\.'
    m = re.match(pattern, last_line, re.DOTALL)
    move_number = 0
    if m is not None:
      game_move = int(m.group(1))

      pattern = r'([+\w-]+)\s*(?:{.*?})?\s*(?:\.\.\s*|$)'
      matches = re.findall(pattern, last_line, re.DOTALL)
      if matches is not None:
        move_number = 4 * (game_move - 1) + len(matches)
        last_move = _standardize_move(matches[-1])

    # Determine if SelfPartner
    is_self_partner = False
    if 'SelfPartner' in pgn4:
      is_self_partner = True

    my_colors = set()

    # Helper to check if TeamEnigma owns a specific color header
    def check_header(color_name):
      pattern = f'\\[{color_name} "TeamEnigma'
      return bool(re.search(pattern, pgn4, re.IGNORECASE))

    if check_header('Red'):
      my_colors.add('r')
      if is_self_partner:
        my_colors.add('y')
    
    if check_header('Blue'):
      my_colors.add('b')
      if is_self_partner:
        my_colors.add('g')

    if not is_self_partner:
      if check_header('Yellow'):
        my_colors.add('y')
      if check_header('Green'):
        my_colors.add('g')
    
    my_colors_list = list(my_colors)

    # Maintain legacy team string for UCI/Eval purposes
    team = 'no_team'
    if 'r' in my_colors or 'y' in my_colors:
      team = 'red_yellow'
    elif 'b' in my_colors or 'g' in my_colors:
      team = 'blue_green'

    return Pgn4Info(base_time_ms, incr_time_ms, delay_time_ms, last_move, team,
        move_number, game_number, my_colors_list)


class Server:

  def __init__(self):
    self._token = _read_api_token(_API_KEY_FILENAME)
    self._uci = uci_wrapper.UciWrapper(
        args.num_threads, 
        args.max_depth,
        args.ponder,
        enable_nnue=args.enable_nnue,
        nnue_path=args.nnue_path
    )
    self._api = api.Api(_SERVER_URL, self._token, _BOT_NAME, _BOT_VERSION)
    self._pgn4_info = None
    self._last_arrow_request = None
    self._gameoverchat = False
    self._game_number = None

  def _handle_gameover(self):
    if not self._gameoverchat:
      self._gameoverchat = True
      self._api.chat('gg')
      self._uci.maybe_stop_ponder_thread()

  def _get_clock_from_pgn(self, pgn: str, active_color: str, base_time_ms: int) -> float:
      """Extracts the last clock time for the active color from the PGN."""
      color_map = {'r': 0, 'b': 1, 'y': 2, 'g': 3}
      
      if active_color not in color_map:
          return float(base_time_ms)
          
      target_idx = color_map[active_color]
      matches = re.findall(r'clock=(\d+)', pgn)
      
      for i in range(len(matches) - 1, -1, -1):
          if i % 4 == target_idx:
              return float(matches[i])
              
      return float(base_time_ms)

  def _handle_stream_json(self, json_response):
    if 'pgn4' in json_response:
      pgn4_info = Pgn4Info.FromString(json_response['pgn4'])
      if pgn4_info is not None:
        self._pgn4_info = pgn4_info
        if args.asymmetric_eval:
          self._uci.set_team(pgn4_info.team)
        if (pgn4_info.game_number is not None
            and self._game_number != pgn4_info.game_number):
          self._gameoverchat = False
          self._game_number = pgn4_info.game_number

    info = json_response.get('info')
    if info:
      info = info.lower()
      if 'game starting' in info or 'no game found' in info:
        return False

    fen = None
    if 'fen4' in json_response:
      fen = json_response['fen4']
    elif 'move' in json_response and 'fen' in json_response['move']:
      fen = json_response['move']['fen']

    if fen:
        fen = fen.replace('\n', '')

        if 'gameOver' in fen:
          if args.arrows:
            self.clear_arrows()
          self._handle_gameover()
          return True
        
        if fen == '4PCo':
          fen = uci_wrapper.START_FEN_OLD
        elif fen == '4PC':
          fen = uci_wrapper.START_FEN_NEW
        elif fen == '4PCb':
          fen = uci_wrapper.START_FEN_BY
        elif fen == '4PCn':
          fen = uci_wrapper.START_FEN_BYG

        # Make sure it's our turn based on active color
        is_my_turn = False
        active_color = fen.split('-')[0].lower()
        
        if self._pgn4_info and self._pgn4_info.my_colors:
             if active_color in self._pgn4_info.my_colors:
                 is_my_turn = True
        
        if not is_my_turn:
            return False

        move = None
        if args.enable_tablebase:
          move = tablebase.FEN_TO_BEST_MOVE.get(fen)
        score = None
        
        if move is None:
          self._uci.set_position(fen)

          # --- TIME MANAGEMENT LOGIC ---
          assert self._pgn4_info is not None
          
          base_time_ms = self._pgn4_info.base_time_ms
          incr_ms = self._pgn4_info.incr_time_ms
          delay_ms = self._pgn4_info.delay_time_ms
          move_time_ms = 0.0
          
          clock_ms = None
          if 'clock' in json_response:
              clock_ms = float(json_response['clock'])
          elif 'move' in json_response and 'clock' in json_response['move']:
              clock_ms = float(json_response['move']['clock'])
          elif 'pgn4' in json_response:
              clock_ms = self._get_clock_from_pgn(json_response['pgn4'], active_color, base_time_ms)

          if clock_ms is not None:
              is_in_time_pressure = clock_ms < (base_time_ms * _TIME_PRESSURE_THRESHOLD_PERCENT)
              time_pressure_multiplier = _TIME_PRESSURE_FACTOR_DECREASE if is_in_time_pressure else 1.0

              if delay_ms > 0:
                  # Full delay + capped base time; no pressure multiplier as delay doesn't accumulate
                  move_time_ms = float(delay_ms)
                  if not is_in_time_pressure:
                      move_time_ms += min(clock_ms / 40.0, delay_ms * 0.5)
                  else:
                      print("[TIME] Low on time! Sticking to delay buffer. ", end='')
              elif incr_ms > 0:
                  # Standard increment math
                  move_time_ms = (incr_ms + (clock_ms / 20.0)) * time_pressure_multiplier
              else:
                  # Sudden death exponential scaling
                  moves_played = self._pgn4_info.played_n_moves
                  divisor = 50 * (1.01 ** moves_played)
                  move_time_ms = (clock_ms / divisor) * time_pressure_multiplier
              
              print(f"[TIME] Clock: {clock_ms/1000:.1f}s. ", end='')
          else:
              # Missing clock fallback logic
              if delay_ms > 0:
                  move_time_ms = delay_ms * 0.90
              elif incr_ms > 0:
                  move_time_ms = float(incr_ms)
              else:
                  move_time_ms = _MIN_MOVE_TIME_MS
              print("[TIME] (Clock missing). ", end='')

          # Apply safety margin and bounds
          move_time_ms *= _SAFETY_MARGIN
          if args.play_fast:
              move_time_ms *= 0.30
          
          final_move_time_ms = int(min(max(move_time_ms, _MIN_MOVE_TIME_MS), _MAX_MOVE_MS))
          print(f"Thinking for: {final_move_time_ms/1000:.2f}s")
          # --- END OF TIME MANAGEMENT LOGIC ---

          res = self._uci.get_best_move(
              final_move_time_ms,
              gameover_callback=self._handle_gameover,
              pv_callback=self.display_arrows,
              last_move=self._pgn4_info.last_move)
          if res.get('gameover'):
            self._handle_gameover()
            return True
          move = res['best_move']
          score = res['score']
          depth = res['depth']

          # Guarded eval chat strictly for team formats
          if args.chat_eval and score is not None:
            team = self._pgn4_info.team
            if team in ('red_yellow', 'blue_green'):
                score_ry = score / 100 if team == 'red_yellow' else -score / 100
                if res.get('ponder_hit', False):
                  score_ry = -score_ry
                self._api.chat(f'eval: {score_ry:.02f}, depth: {depth}')

        self._api.play(move)
        if args.ponder:
          self._uci.ponder(fen, move, self._handle_gameover)

    return False

  def clear_arrows(self):
    self._api.arrow('clear')

  def display_arrows(self, pv: list[str]):
    if not args.arrows:
      return
    parts = ['clear']
    for move in pv[:4]:
      parts.append(f'{move}-50')
    request = ','.join(parts)
    if request == self._last_arrow_request:
      return
    self._last_arrow_request = request
    self._api.arrow(request)

  def run(self):
    print(f"Connecting to {_SERVER_URL} via polling...")
    log_file = "server_dump.txt"
    last_logged_str = None

    while True:
      try:
        response = self._api.get_state()
        if response:
            try:
                current_str = json.dumps(response)
                with open(log_file, "a", encoding="utf-8") as f:
                    if current_str == last_logged_str:
                        f.write('.')
                    else:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        if last_logged_str is not None: f.write("\n")
                        f.write(f"[{timestamp}] {current_str}")
                        last_logged_str = current_str
            except Exception as log_error:
                print(f"Could not write to log file: {log_error}")
            self._handle_stream_json(response)
      except Exception as e:
        print(f"Error in poll loop: {e}")
      time.sleep(0.1)

if __name__ == '__main__':
  Server().run()