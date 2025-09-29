import re
import argparse
import skopt
import subprocess

from skopt import Optimizer

parser = argparse.ArgumentParser(
    prog='Tuner',
    description='Tunes parameters of the chess program')
parser.add_argument('-tune_params', '--tune_params', type=str,
    required=False, default=None)
parser.add_argument('-scm_path', '--scm_path', type=str, required=True)
parser.add_argument('-engine_path', '--engine_path', type=str, required=True)
parser.add_argument('-fens_path', '--fens_path', type=str, required=False,
    default='')
parser.add_argument('-n_calls', '--n_calls', type=int,
    required=False, default=100)
args = parser.parse_args()

# NOTE: These params are outdated and we don't use this file (yet) to tune any
# hyper-params of the program.
_PARAMS = {
  'piece_eval_pawn': { 'space': skopt.space.space.Integer(10, 300), },
  'piece_eval_knight': { 'space': skopt.space.space.Integer(100, 600), },
  'piece_eval_bishop': { 'space': skopt.space.space.Integer(200, 600), },
  'piece_eval_rook': { 'space': skopt.space.space.Integer(200, 900), },
  'piece_eval_queen': { 'space': skopt.space.space.Integer(500, 1500), },
}



def tune_params(param_names):
  dimensions = []
  for param_name in param_names:
    if param_name not in _PARAMS:
      raise ValueError(
          'Parameter optimization not handled for parameter: ' + param_name)
    param_info = _PARAMS[param_name]
    dimensions.append(param_info['space'])

  def format_param_values(param_values):
    param_str = [f'{name}={value}' for name, value in zip(param_names, param_values)]
    param_str = ', '.join(param_str)
    return f'[{param_str}]'

  def f(param_values):
    assert len(param_names) == len(param_values)
    custom_args = []
    for param_name, param_value in zip(param_names, param_values):
      custom_args.extend([
          '--custom2', f'"setoption name {param_name} value {param_value}"',
      ])

    scm_args = [
      args.scm_path,
      '--e1', args.engine_path,
      '--e2', args.engine_path,
      '--fixed', '200',
      '--games', '100',
      '--threads', '10',
      '--maxmoves', '100',
    ] + custom_args
    if args.fens_path:
      scm_args.extend(['--fens', args.fens_path])
    full_cmd = ' '.join(scm_args)

    result = subprocess.run(scm_args, capture_output=True, encoding='utf-8')
    output = result.stdout

    lines = output.split('\n')
    for line in reversed(lines):
      # Use a raw string (r'...') for the pattern to avoid SyntaxWarning
      pattern = r'Engine1.*?(\d+) wins.*?Engine2.*?(\d+) wins'
      match = re.search(pattern, line)
      if match is not None:
        win1 = float(match.group(1))
        win2 = float(match.group(2))
        
        games_target_index = scm_args.index('--games') + 1
        games_target = int(scm_args[games_target_index])
        min_games_threshold = 0.9 * games_target

        total_decisive_games = win1 + win2
        if total_decisive_games < min_games_threshold:
            print(f"WARNING: Incomplete match detected. Decisive games played ({total_decisive_games}) is less than 90% threshold ({min_games_threshold}).")
            print("         This data point will be ignored by the optimizer.")
            print('         Params:', format_param_values(param_values))
            return None # Return None to signal a failed run

        print('Baseline Wins:', win1, 'Tuned Wins:', win2, 'param_values:', format_param_values(param_values))
        
        if total_decisive_games == 0:
          return 0.5
        
        loss_rate = win1 / total_decisive_games
        return loss_rate
        
    print('Error: Could not parse match results from SCM output.')
    print('Output:\n', output)
    return 1.0

  # --- MANUAL OPTIMIZATION LOOP ---
  # We replace skopt.gp_minimize with our own loop to handle failed runs.

  # 1. Initialize the optimizer
  optimizer = Optimizer(
      dimensions=dimensions,
      random_state=1, # for reproducibility
      base_estimator="GP", # Gaussian Process
      acq_func="gp_hedge", # A good default acquisition function
      n_initial_points=10, # Number of random points to sample before using the model
  )

  # 2. Run the optimization loop until we have n_calls SUCCESSFUL results
  successful_calls = 0
  result = None
  while successful_calls < args.n_calls:
      print(f"\n--- Starting Iteration {successful_calls + 1}/{args.n_calls} ---")
      
      # Ask the optimizer for the next point to evaluate
      next_x = optimizer.ask()
      
      # Run our objective function
      next_y = f(next_x)
      
      # If the function returned a valid score (not None), tell the optimizer.
      # Otherwise, we do nothing, effectively skipping this iteration.
      if next_y is not None:
          result = optimizer.tell(next_x, next_y)
          successful_calls += 1
          
          # Manually call a "callback" to print progress
          win_rate = 1.0 - result.fun if result.fun is not None else 0.0
          print(f'#iters: {successful_calls}, Best Win Rate So Far: {win_rate:.2%}, Current Params: {format_param_values(result.x)}')
      else:
          print("Iteration failed. Asking optimizer for a new point.")
  
  # --- END: MANUAL OPTIMIZATION LOOP ---

  import numpy as np

  print('\n===== Tuning Complete: Final Analysis =====')
  if result:
    # --- 1. Best OBSERVED Parameters (potentially lucky) ---
    # This is the single set of parameters that achieved the best score in one iteration.
    print('--- Best Observed Run (most "lucky" iteration) ---')
    best_observed_idx = np.argmin(result.func_vals)
    best_observed_params = result.x_iters[best_observed_idx]
    best_observed_score = result.func_vals[best_observed_idx]

    for param_name, best_value in zip(param_names, best_observed_params):
      print(f'{param_name} = {best_value}')
    print(f'Observed win rate in its single match: {1.0 - best_observed_score:.2%}\n')


    # --- 2. Best PREDICTED Parameters (most robust) ---
    # This is the set of parameters the GP model predicts is the best after
    # considering ALL data points. This is the value stored in `result.x`.
    # THIS IS THE ONE WE SHOULD GENERALLY USE.
    print('--- Best Predicted Parameters (Optimizer\'s Final Recommendation) ---')
    best_predicted_params = result.x
    predicted_score = result.fun

    for param_name, best_value in zip(param_names, best_predicted_params):
      print(f'{param_name} = {best_value}')
    print(f'This set is the model\'s best estimate for achieving a win rate of: {1.0 - predicted_score:.2%}')
    print('(This is the most reliable result as it uses data from all iterations to smooth out noise.)')

  else:
    print("No successful iterations were completed.")


def main():
  param_names = args.tune_params.split(',')
  print('Tuning params:', param_names)
  tune_params(param_names)
    

if __name__ == '__main__':
  main()