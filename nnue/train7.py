# train7.py
# MODIFIED: Ensure training and validation data come from separate games.
import argparse
import gc
import glob
import matplotlib.pyplot as plt # Not used in provided snippet, but often for plotting history
import numpy as np
import os
import shutil
import subprocess
import tensorflow as tf
import time
from collections import deque # Added for global validation pool

# --- Global Validation Pool Components ---
g_validation_pool_boards = deque()
g_validation_pool_scores = deque()
g_validation_pool_per_depth_scores = deque()


# --- Argument Parser ---
parser = argparse.ArgumentParser(
    prog='NNUE trainer',
    description='Trains NNUE using self-play and moderate depth search')

parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--num_self_play_loops', required=False, default=10, type=int)
parser.add_argument('--train_positions_per_loop', required=False, default=100000, type=int)
parser.add_argument('--validation_fraction', required=False, default=0.1, type=float, help="Fraction of train_positions_per_loop to generate as separate validation games each loop.")
parser.add_argument('--train_last_n', required=False, default=3, type=int, help="Number of last N generations of training data to use.")
parser.add_argument('--val_pool_keep_last_n_gens', required=False, default=3, type=int, help="Number of last N generations of validation data to keep in the pool.")
parser.add_argument('--search_depth', required=False, default=8, type=int)
parser.add_argument('--epochs_per_loop', required=False, default=50, type=int)
parser.add_argument('--batch_size', required=False, default=256, type=int)
parser.add_argument('--nnue_search_rate', required=False, default=0.5, type=float)
parser.add_argument('--gen_data_threads', required=False, default=12, type=int, help="Number of threads for gen_data.")
parser.add_argument('--mode', required=False, default='train-new', type=str, choices=['train-new', 'train-existing'])
parser.add_argument('--existing_data_dirs', required=False, type=str)
# --- End Argument Parser ---


@tf.keras.utils.register_keras_serializable()
def one_hot_layer_fn(board_tensor):
    one_hot = tf.one_hot(board_tensor, depth=7, axis=-1)
    s = tf.shape(one_hot)
    batch_dim = s[0]
    output = tf.reshape(one_hot, [batch_dim, 4, 14*14*7])
    return output


def generate_training_data(
    output_dir_absolute, search_depth, num_threads, num_positions,
    nnue_weights_filepath_absolute, nnue_search_rate):
  print(f'generating data ({num_positions} positions) into {output_dir_absolute} using {num_threads} threads...')
  
  script_dir = os.path.dirname(os.path.abspath(__file__))
  workspace_root = os.path.dirname(script_dir)
  prog_filepath = os.path.join(workspace_root, 'bazel-bin', 'nnue', 'gen_data')
  if os.name == 'nt' and not prog_filepath.lower().endswith('.exe'):
      prog_filepath += '.exe'

  args_list = [
    output_dir_absolute,
    str(search_depth),
    str(num_threads),
    str(num_positions),
    str(nnue_search_rate),
  ]
  if nnue_weights_filepath_absolute and os.path.exists(nnue_weights_filepath_absolute):
    args_list.append(nnue_weights_filepath_absolute)
  else:
    if nnue_search_rate > 0.0:
        print(f"Warning: NNUE search rate > 0 but NNUE weights path for gen_data not provided or invalid: {nnue_weights_filepath_absolute}. Gen_data will run without NNUE.")

  final_args = [prog_filepath] + args_list
  print('cmd:', ' '.join(final_args))
  completion = subprocess.run(final_args, capture_output=True, text=True)
  # Only print stdout/stderr if they are not empty, to reduce log spam if gen_data is quiet on success
  if completion.stdout.strip():
    print("gen_data stdout:\n", completion.stdout)
  if completion.stderr.strip():
      print("gen_data stderr:\n", completion.stderr)

  if completion.returncode != 0:
    raise ValueError(
        f'Training data gen finished with nonzero return code: '
        f'{completion.returncode}')


def data_generator(data_dirs_to_load_from, immediate=False, search_depth_for_signature=None): # Renamed arg for clarity
  def gen():
    data_dirs_list = data_dirs_to_load_from # Use the more descriptive name
    if not isinstance(data_dirs_list, list):
      data_dirs_list = [data_dirs_list]
    
    all_board_filepaths = []
    all_score_filepaths = []
    all_per_depth_score_filepaths = []

    for data_dir in data_dirs_list:
        current_dir_board_fps = sorted(glob.glob(os.path.join(data_dir, 'board*.csv')))
        current_dir_score_fps = sorted(glob.glob(os.path.join(data_dir, 'score*.csv')))
        current_dir_pds_fps = sorted(glob.glob(os.path.join(data_dir, 'per_depth_score*.csv')))
        
        if not current_dir_board_fps:
            print(f"Warning: No board files (board*.csv) found in {data_dir}. Skipping this directory.")
            continue

        if not (len(current_dir_board_fps) == len(current_dir_score_fps) == len(current_dir_pds_fps)):
            print(f"Warning: Mismatch in number of data file types found in {data_dir}. "
                  f"Boards: {len(current_dir_board_fps)} ({[os.path.basename(f) for f in current_dir_board_fps]}), "
                  f"Scores: {len(current_dir_score_fps)} ({[os.path.basename(f) for f in current_dir_score_fps]}), "
                  f"PDS: {len(current_dir_pds_fps)} ({[os.path.basename(f) for f in current_dir_pds_fps]}). "
                  f"Skipping this directory.")
            continue
        
        all_board_filepaths.extend(current_dir_board_fps)
        all_score_filepaths.extend(current_dir_score_fps)
        all_per_depth_score_filepaths.extend(current_dir_pds_fps)
    
    if not all_board_filepaths:
        print("Error: No data files found by data_generator after checking all provided directories.")
        return

    for board_filepath, score_filepath, per_depth_score_filepath in zip(
        all_board_filepaths, all_score_filepaths, all_per_depth_score_filepaths):
      print('Loading data from:', board_filepath, flush=True)
      try:
        board_np = np.genfromtxt(board_filepath, delimiter=',', dtype=np.int32)
        score_np = np.genfromtxt(score_filepath, delimiter=',', dtype=np.int32)
        per_depth_score_np = np.genfromtxt(per_depth_score_filepath, delimiter=',', dtype=np.int32)

        if score_np.ndim == 0: score_np = np.expand_dims(score_np, axis=0)
        if board_np.ndim == 1: board_np = np.expand_dims(board_np, axis=0)
        if per_depth_score_np.ndim == 1: per_depth_score_np = np.expand_dims(per_depth_score_np, axis=0)
        
        if board_np.shape[0] == 0:
            print(f"Warning: File {board_filepath} is empty or contains no parsable data. Skipping.")
            continue

        if not (board_np.shape[0] == score_np.shape[0] == per_depth_score_np.shape[0]):
            print(f"Warning: Mismatch in number of samples in file triplet starting with {board_filepath}")
            print(f"  Board shape: {board_np.shape}, Score shape: {score_np.shape}, PDS shape: {per_depth_score_np.shape}")
            continue 

        yield (tf.convert_to_tensor(board_np, dtype=tf.int32),
               tf.convert_to_tensor(score_np, dtype=tf.int32),
               tf.convert_to_tensor(per_depth_score_np, dtype=tf.int32))
      except Exception as e:
        print(f"Error loading or processing file {board_filepath}: {e}")
        continue
      gc.collect()

  if not immediate:
    return gen 

  results_list = list(gen()) 
  if not results_list:
      raise ValueError("No data could be loaded by the data generator (immediate=True). Check file paths, formats, and content.")
  def immediate_gen():
    for x in results_list:
      yield x
  
  return immediate_gen

def convert_scores_to_probs(scores):
  denom = -10.0 / tf.math.log(1.0/.9 - 1.0)
  score = tf.cast(scores, tf.float32) / 100.0 / denom
  return tf.math.sigmoid(score)


def convert_probs_to_scores(probs):
  epsilon = 1e-8
  prob = np.maximum(epsilon, np.minimum(1.0 - epsilon, probs))
  d = -10.0 / np.log(1.0/.9 - 1.0)
  return 100.0 * d * np.log(prob/(1.0 - prob))

def create_dataset_from_np(boards_data, scores_data, batch_size_arg):
    dataset = tf.data.Dataset.from_tensor_slices((boards_data, scores_data))
    
    def map_fn_split(board, score):
        board_reshaped = tf.reshape(board, [4, 14*14])
        probs = convert_scores_to_probs(score)
        probs_expanded = tf.expand_dims(probs, axis=-1) 
        return board_reshaped, probs_expanded

    dataset = dataset.map(map_fn_split, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size_arg)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_model(train_data_dirs, 
                val_boards_np, val_scores_np, val_pds_np, 
                last_model_weights_filepath, model_save_dir,
                epochs_per_loop, batch_size_arg, search_depth_arg):
  print('Training model...')
  
  num_depth_cols = search_depth_arg + 1

  train_gen_callable = data_generator(train_data_dirs, immediate=False, search_depth_for_signature=search_depth_arg)

  train_dataset = tf.data.Dataset.from_generator(
      train_gen_callable,
      output_signature=(
        tf.TensorSpec(shape=(None, 784), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None, num_depth_cols), dtype=tf.int32)
      )
  )
  
  def map_fn_train(board, score, per_depth_score):
    board = tf.reshape(board, [-1, 4, 14*14])
    probs = convert_scores_to_probs(score)
    probs = tf.expand_dims(probs, axis=-1)
    return board, probs

  train_dataset = train_dataset.map(map_fn_train, num_parallel_calls=tf.data.AUTOTUNE)
  train_dataset = train_dataset.unbatch() 
  
  # Estimate total training samples for shuffle buffer
  # Assuming each dir in train_data_dirs contains roughly args.train_positions_per_loop samples
  # This is an approximation for 'train_new' mode. For 'train_existing', it's just one dir.
  num_train_dirs_for_shuffle = len(train_data_dirs) if isinstance(train_data_dirs, list) else 1
  estimated_train_samples_for_shuffle = num_train_dirs_for_shuffle * args.train_positions_per_loop
  shuffle_buffer_size = min(estimated_train_samples_for_shuffle, 200000) 
  print(f"Using shuffle buffer size for training: {shuffle_buffer_size}")
  if shuffle_buffer_size > 0 :
      train_dataset = train_dataset.shuffle(shuffle_buffer_size)

  train_dataset = train_dataset.batch(batch_size_arg)
  train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

  val_dataset_for_keras = None
  if val_boards_np is not None and val_scores_np is not None and len(val_boards_np) > 0:
      if len(val_boards_np) >= batch_size_arg: 
          print(f"Preparing validation dataset with {len(val_boards_np)} samples.")
          val_dataset_for_keras = create_dataset_from_np(val_boards_np, val_scores_np, batch_size_arg)
      else:
          print(f"Validation pool has {len(val_boards_np)} samples, less than batch size {batch_size_arg}. Skipping Keras validation.")
  else:
      print("Validation pool is empty. Skipping Keras validation.")

  model = None
  if last_model_weights_filepath and os.path.exists(last_model_weights_filepath):
    print(f'Reloading entire model (including optimizer state) from: {last_model_weights_filepath}')
    try:
        model = tf.keras.models.load_model(last_model_weights_filepath)
        model.compile(
            optimizer='adam', 
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
        print("Model recompiled after loading.")
    except Exception as e:
        print(f"ERROR: Could not load model from {last_model_weights_filepath}: {e}. Training from scratch.")
  
  if model is None: 
    print("Creating a new model.")
    model = tf.keras.Sequential(layers=[
      tf.keras.Input(shape=(4, 14*14), dtype=tf.int32, name="board_input"),
      tf.keras.layers.Lambda(one_hot_layer_fn, output_shape=(4, 14*14*7)),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
      optimizer='adam',
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
  
  model.summary()
  gc.collect()

  print("Starting model.fit...")
  history = model.fit(
      train_dataset, 
      epochs=epochs_per_loop, 
      verbose=1,
      validation_data=val_dataset_for_keras
  )

  if history.history and 'mean_absolute_error' in history.history and history.history['mean_absolute_error']:
    print('Training MAE on probabilities (last epoch):', history.history['mean_absolute_error'][-1])
  if history.history and 'val_mean_absolute_error' in history.history and history.history['val_mean_absolute_error']:
    print('Validation MAE on probabilities (last epoch):', history.history['val_mean_absolute_error'][-1])

  os.makedirs(model_save_dir, exist_ok=True)
  model.save(os.path.join(model_save_dir, 'nnue.h5'), save_format='h5')

  def save_weights(filename, np_arr):
    full_filepath = os.path.join(model_save_dir, filename)
    np.savetxt(full_filepath, np_arr.reshape(-1, np_arr.shape[-1] if np_arr.ndim > 1 else np_arr.size), delimiter=',', fmt='%f')

  dense_layers = [x for x in model.layers if isinstance(x, tf.keras.layers.Dense)]
  for i, layer in enumerate(dense_layers):
    weights = layer.get_weights() 
    if len(weights) == 2:
        kernel, bias = weights
        save_weights(f'layer_{i}.kernel', kernel); save_weights(f'layer_{i}.bias', bias)     
    elif weights: kernel = weights[0]; save_weights(f'layer_{i}.kernel', kernel)

  print("\n--- Final Evaluation on Global Validation Set ---")
  if val_boards_np is not None and val_scores_np is not None and val_pds_np is not None and len(val_boards_np) > 0:
    val_boards_reshaped_for_pred = val_boards_np.reshape((-1, 4, 14*14))
    non_mate_threshold = 10000 
    fltr_val = np.abs(val_scores_np) < non_mate_threshold
    
    val_brd_f = val_boards_reshaped_for_pred[fltr_val]
    val_scr_f = val_scores_np[fltr_val]

    if val_brd_f.shape[0] > 0:
      pred_probs_val_f = model.predict(val_brd_f, batch_size=batch_size_arg)
      pred_scores_val_f = convert_probs_to_scores(pred_probs_val_f.flatten())
      val_mae_centipawns = np.mean(np.abs(val_scr_f - pred_scores_val_f))
      print(f'VALIDATION MAE (model scores vs search scores, non-extreme): {val_mae_centipawns:.2f} cp')
    else:
      print("No non-extreme validation positions for score MAE analysis.")

    print('Per-depth score analysis (on VALIDATION data):')
    val_pds_fltr = val_pds_np[fltr_val] 
    if val_pds_fltr.shape[0] > 0 and val_pds_fltr.shape[1] > 1:
        final_depth_scores_val = val_pds_fltr[:, -1]
        for i in range(val_pds_fltr.shape[1] - 1):
            initial_depth_scores_val = val_pds_fltr[:, i]
            mae_depth_vs_final_val = np.mean(np.abs(initial_depth_scores_val - final_depth_scores_val))
            print(f'  VALIDATION MAE (depth {i} vs final depth): {mae_depth_vs_final_val:.2f} cp')
    else:
        print("Not enough validation data or depths for per-depth score analysis.")
  else:
    print("Global validation pool is empty. Skipping final evaluation stats.")


def train_new(args):
  output_dir_relative = args.output_dir
  output_dir_absolute = os.path.abspath(output_dir_relative)

  model_dir_name = 'models'
  gen_model_dir_name = 'gen_models'
  current_model_working_dir = os.path.join(output_dir_absolute, model_dir_name)
  generation_models_archive_path = os.path.join(output_dir_absolute, gen_model_dir_name)

  os.makedirs(current_model_working_dir, exist_ok=True)
  os.makedirs(generation_models_archive_path, exist_ok=True)

  gen_id = 1
  while True:
      archived_model_path_gen_dir = os.path.join(generation_models_archive_path, f'gen_{gen_id}')
      # Check if the directory itself exists, not just nnue.h5, as we save other files too.
      if os.path.exists(archived_model_path_gen_dir):
          gen_id += 1
      else:
          break
  
  last_model_weights_for_training = None
  nnue_weights_dir_for_gen_data = None
  if gen_id > 1:
      prev_gen_archive_dir = os.path.join(generation_models_archive_path, f'gen_{gen_id-1}')
      prev_gen_archived_model_h5_path = os.path.join(prev_gen_archive_dir, 'nnue.h5')
      if os.path.exists(prev_gen_archived_model_h5_path):
          last_model_weights_for_training = prev_gen_archived_model_h5_path
          nnue_weights_dir_for_gen_data = prev_gen_archive_dir # This is the directory for C++ weights
      else:
          print(f"Warning: Resuming at gen {gen_id}, but prev gen_{gen_id-1} model .h5 or directory not found.")
  
  print(f"Starting/Resuming at generation {gen_id}. Prev Keras model for training: {last_model_weights_for_training}")
  print(f"Prev C++ NNUE weights dir for gen_data: {nnue_weights_dir_for_gen_data}")

  while gen_id <= args.num_self_play_loops:
    print(f'\n--- Loop (Generation): {gen_id} ---')
    
    weights_for_gen_data_cmd = None
    if args.nnue_search_rate > 0.0 and nnue_weights_dir_for_gen_data and os.path.exists(nnue_weights_dir_for_gen_data):
        weights_for_gen_data_cmd = nnue_weights_dir_for_gen_data
    elif args.nnue_search_rate > 0.0:
        print(f"Note: nnue_search_rate > 0 but no prev NNUE weights dir for gen_data. Running gen_data without NNUE eval.")

    # --- 1. Generate and Process TRAINING Data ---
    current_gen_raw_train_data_dir = os.path.join(output_dir_absolute, f'raw_train_gen_{gen_id}')
    if os.path.exists(current_gen_raw_train_data_dir): shutil.rmtree(current_gen_raw_train_data_dir)
    os.makedirs(current_gen_raw_train_data_dir, exist_ok=True)

    generate_training_data(
        current_gen_raw_train_data_dir,
        args.search_depth,
        args.gen_data_threads,
        args.train_positions_per_loop,
        nnue_weights_filepath_absolute=weights_for_gen_data_cmd,
        nnue_search_rate=args.nnue_search_rate)

    print(f"Loading raw training data from: {current_gen_raw_train_data_dir}")
    train_data_gen_callable = data_generator([current_gen_raw_train_data_dir], immediate=True, search_depth_for_signature=args.search_depth)
    all_train_parts = list(train_data_gen_callable())
    
    if not all_train_parts:
        print(f"Error: No training data loaded from {current_gen_raw_train_data_dir}. Skipping generation {gen_id}.")
        gen_id += 1
        if os.path.exists(current_gen_raw_train_data_dir): shutil.rmtree(current_gen_raw_train_data_dir)
        continue

    raw_train_boards_list, raw_train_scores_list, raw_train_pds_list = zip(*all_train_parts)
    raw_train_boards_np = np.concatenate([b.numpy() for b in raw_train_boards_list], axis=0)
    raw_train_scores_np = np.concatenate([s.numpy() for s in raw_train_scores_list], axis=0)
    raw_train_pds_np    = np.concatenate([p.numpy() for p in raw_train_pds_list], axis=0)
    del raw_train_boards_list, raw_train_scores_list, raw_train_pds_list, all_train_parts
    gc.collect()
    
    print(f"Loaded {raw_train_boards_np.shape[0]} new training samples for gen {gen_id}.")

    # Shuffle training data
    num_new_train_samples = raw_train_boards_np.shape[0]
    shuffled_indices_train = np.random.permutation(num_new_train_samples)
    new_train_boards = raw_train_boards_np[shuffled_indices_train]
    new_train_scores = raw_train_scores_np[shuffled_indices_train]
    new_train_pds    = raw_train_pds_np[shuffled_indices_train]

    # Save 100% of these (shuffled) training samples
    current_gen_train_storage_dir = os.path.join(output_dir_absolute, f'train_data_gen_{gen_id}')
    if os.path.exists(current_gen_train_storage_dir): shutil.rmtree(current_gen_train_storage_dir)
    os.makedirs(current_gen_train_storage_dir, exist_ok=True)
    
    np.savetxt(os.path.join(current_gen_train_storage_dir, 'board_0.csv'), new_train_boards, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(current_gen_train_storage_dir, 'score_0.csv'), new_train_scores, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(current_gen_train_storage_dir, 'per_depth_score_0.csv'), new_train_pds, delimiter=',', fmt='%d')
    print(f"Saved {new_train_boards.shape[0]} training samples to {current_gen_train_storage_dir}")
    if os.path.exists(current_gen_raw_train_data_dir): shutil.rmtree(current_gen_raw_train_data_dir)


    # --- 2. Generate and Process VALIDATION Data ---
    validation_positions_this_loop = int(args.train_positions_per_loop * args.validation_fraction)
    if validation_positions_this_loop > 0:
        current_gen_raw_val_data_dir = os.path.join(output_dir_absolute, f'raw_val_gen_{gen_id}')
        if os.path.exists(current_gen_raw_val_data_dir): shutil.rmtree(current_gen_raw_val_data_dir)
        os.makedirs(current_gen_raw_val_data_dir, exist_ok=True)

        generate_training_data(
            current_gen_raw_val_data_dir,
            args.search_depth,
            args.gen_data_threads,
            validation_positions_this_loop, # Separate amount for validation
            nnue_weights_filepath_absolute=weights_for_gen_data_cmd, # Use same model for consistency
            nnue_search_rate=args.nnue_search_rate
        )
        print(f"Loading raw validation data from: {current_gen_raw_val_data_dir}")
        val_data_gen_callable = data_generator([current_gen_raw_val_data_dir], immediate=True, search_depth_for_signature=args.search_depth)
        all_val_parts = list(val_data_gen_callable())

        if all_val_parts:
            raw_val_boards_list, raw_val_scores_list, raw_val_pds_list = zip(*all_val_parts)
            raw_val_boards_np = np.concatenate([b.numpy() for b in raw_val_boards_list], axis=0)
            raw_val_scores_np = np.concatenate([s.numpy() for s in raw_val_scores_list], axis=0)
            raw_val_pds_np    = np.concatenate([p.numpy() for p in raw_val_pds_list], axis=0)
            del raw_val_boards_list, raw_val_scores_list, raw_val_pds_list, all_val_parts
            gc.collect()

            print(f"Loaded {raw_val_boards_np.shape[0]} new validation samples for gen {gen_id}.")

            # Shuffle validation data before adding to pool
            num_new_val_samples = raw_val_boards_np.shape[0]
            shuffled_indices_val = np.random.permutation(num_new_val_samples)
            new_val_boards = raw_val_boards_np[shuffled_indices_val]
            new_val_scores = raw_val_scores_np[shuffled_indices_val]
            new_val_pds    = raw_val_pds_np[shuffled_indices_val]

            for i in range(num_new_val_samples):
                g_validation_pool_boards.append(new_val_boards[i])
                g_validation_pool_scores.append(new_val_scores[i])
                g_validation_pool_per_depth_scores.append(new_val_pds[i])
            print(f"Added {num_new_val_samples} samples to global validation pool. Pool size: {len(g_validation_pool_boards)}")
        else:
            print(f"Warning: No validation data loaded from {current_gen_raw_val_data_dir}.")
        
        if os.path.exists(current_gen_raw_val_data_dir): shutil.rmtree(current_gen_raw_val_data_dir)
    else:
        print("Skipping generation of new validation data as validation_positions_this_loop is 0.")


    # --- 3. Manage Global Validation Pool Size ---
    # Max size is based on num_validation_generations_to_keep * (average) validation_positions_per_loop
    max_validation_pool_target_size = int(args.val_pool_keep_last_n_gens * validation_positions_this_loop) 
    
    if max_validation_pool_target_size > 0: # Only manage if there's a target size
        while len(g_validation_pool_boards) > max_validation_pool_target_size:
            g_validation_pool_boards.popleft()
            g_validation_pool_scores.popleft()
            g_validation_pool_per_depth_scores.popleft()
    print(f"Managed global validation pool. New size: {len(g_validation_pool_boards)} (max target: {max_validation_pool_target_size})")


    # --- 4. Prepare data for train_model ---
    training_data_source_dirs_for_model = []
    start_data_idx = max(1, gen_id - args.train_last_n + 1)
    for i in range(start_data_idx, gen_id + 1):
        data_source_dir = os.path.join(output_dir_absolute, f'train_data_gen_{i}')
        if os.path.exists(data_source_dir):
            training_data_source_dirs_for_model.append(data_source_dir)
        else:
            print(f"Warning: Training data directory {data_source_dir} not found for training sources.")
    
    if not training_data_source_dirs_for_model:
        print("Error: No training data sources found. Skipping training for this generation.")
        gen_id +=1
        continue

    print(f"Training with data from: {training_data_source_dirs_for_model}")

    val_boards_np_for_model = np.array(list(g_validation_pool_boards)) if g_validation_pool_boards else None
    val_scores_np_for_model = np.array(list(g_validation_pool_scores)) if g_validation_pool_scores else None
    val_pds_np_for_model    = np.array(list(g_validation_pool_per_depth_scores)) if g_validation_pool_per_depth_scores else None

    train_model(
        training_data_source_dirs_for_model, 
        val_boards_np_for_model, val_scores_np_for_model, val_pds_np_for_model,
        last_model_weights_for_training,
        current_model_working_dir, # This is '.../output_dir/models'
        args.epochs_per_loop,
        args.batch_size,
        args.search_depth
    )

    archive_current_gen_model_to = os.path.join(generation_models_archive_path, f'gen_{gen_id}')
    if os.path.exists(archive_current_gen_model_to): shutil.rmtree(archive_current_gen_model_to)
    shutil.copytree(current_model_working_dir, archive_current_gen_model_to) # Copy '.../models' to '.../gen_models/gen_X'
    print(f"Archived model for generation {gen_id} to: {archive_current_gen_model_to}")

    last_model_weights_for_training = os.path.join(archive_current_gen_model_to, 'nnue.h5')
    nnue_weights_dir_for_gen_data = archive_current_gen_model_to # This is the directory for C++ layer files

    gen_id += 1


def train_existing(args):
  dirs_pattern = args.existing_data_dirs
  if not dirs_pattern:
    raise ValueError('Must provide --existing_data_dirs')
  
  data_dirs = glob.glob(dirs_pattern)
  if not data_dirs:
    raise ValueError(f'No data dirs found for pattern: {dirs_pattern}')

  data_dirs_absolute = [os.path.abspath(d) for d in data_dirs]
  print(f"Training with existing data from: {data_dirs_absolute}")

  output_dir_relative = args.output_dir
  output_dir_absolute = os.path.abspath(output_dir_relative)
  model_save_dir = os.path.join(output_dir_absolute, 'models_from_existing_data')
  os.makedirs(model_save_dir, exist_ok=True)

  start_time = time.time()

  print("Loading all existing data for train/test split...")
  all_data_gen_callable = data_generator(data_dirs_absolute, immediate=True, search_depth_for_signature=args.search_depth)
  all_parts = list(all_data_gen_callable())

  if not all_parts:
      raise ValueError("No data loaded from existing directories.")

  all_boards_list, all_scores_list, all_pds_list = zip(*all_parts)
  all_boards_np = np.concatenate([b.numpy() for b in all_boards_list], axis=0)
  all_scores_np = np.concatenate([s.numpy() for s in all_scores_list], axis=0)
  all_pds_np    = np.concatenate([p.numpy() for p in all_pds_list], axis=0)
  del all_boards_list, all_scores_list, all_pds_list, all_parts
  gc.collect()
  
  dataset_size = all_boards_np.shape[0]
  print(f"Total samples loaded from existing data: {dataset_size}")

  indices = np.random.permutation(dataset_size)
  shuffled_boards_np = all_boards_np[indices]
  shuffled_scores_np = all_scores_np[indices]
  shuffled_pds_np    = all_pds_np[indices]

  val_split = args.validation_fraction # Use the global validation_fraction for consistency
  num_val_samples = int(val_split * dataset_size)
  
  train_boards_np = shuffled_boards_np[num_val_samples:]
  train_scores_np = shuffled_scores_np[num_val_samples:]
  train_pds_np    = shuffled_pds_np[num_val_samples:] # Corresponding PDS for training

  val_boards_np = shuffled_boards_np[:num_val_samples]
  val_scores_np = shuffled_scores_np[:num_val_samples]
  val_pds_np    = shuffled_pds_np[:num_val_samples]
  
  print(f"Training samples: {train_boards_np.shape[0]}, Validation samples: {val_boards_np.shape[0]}")
  
  temp_train_dir = os.path.join(output_dir_absolute, "temp_existing_train_data")
  if os.path.exists(temp_train_dir): shutil.rmtree(temp_train_dir)
  os.makedirs(temp_train_dir)
  np.savetxt(os.path.join(temp_train_dir, 'board_0.csv'), train_boards_np, delimiter=',', fmt='%d')
  np.savetxt(os.path.join(temp_train_dir, 'score_0.csv'), train_scores_np, delimiter=',', fmt='%d')
  np.savetxt(os.path.join(temp_train_dir, 'per_depth_score_0.csv'), train_pds_np, delimiter=',', fmt='%d') 

  train_model(
      [temp_train_dir], 
      val_boards_np, val_scores_np, val_pds_np,
      last_model_weights_filepath=None, 
      model_save_dir=model_save_dir,
      epochs_per_loop=args.epochs_per_loop,
      batch_size_arg=args.batch_size,
      search_depth_arg=args.search_depth
  )
  
  if os.path.exists(temp_train_dir): shutil.rmtree(temp_train_dir)

  duration = time.time() - start_time
  print(f'Completed training on existing data in {duration:.3f} seconds. Model saved to: {model_save_dir}')


def train():
  args_parsed = parser.parse_args()
  global args 
  args = args_parsed

  if args.mode == 'train-new':
    train_new(args)
  else:
    assert args.mode == 'train-existing'
    train_existing(args)


if __name__ == '__main__':
  train()

## To train a new model, use:

# python nnue/train7.py --output_dir=data --num_self_play_loops=100 --train_positions_per_loop=100000 --validation_fraction=0.2 --train_last_n=3 --val_pool_keep_last_n_gens=5 --search_depth=8 --epochs_per_loop=1 --batch_size=256 --nnue_search_rate=0.0 --gen_data_threads=12

# python nnue/train7.py --output_dir=data --num_self_play_loops=100 --train_positions_per_loop=25000 --validation_fraction=0.2 --train_last_n=6 --val_pool_keep_last_n_gens=10 --search_depth=8 --epochs_per_loop=1 --batch_size=256 --nnue_search_rate=0.0 --gen_data_threads=12

# python nnue/train7.py --output_dir=data --num_self_play_loops=100 --train_positions_per_loop=10000 --validation_fraction=0.2 --train_last_n=10 --val_pool_keep_last_n_gens=15 --search_depth=8 --epochs_per_loop=1 --batch_size=256 --nnue_search_rate=0.0 --gen_data_threads=12

# python nnue/train7.py --output_dir=data --num_self_play_loops=100 --train_positions_per_loop=5000 --validation_fraction=0.2 --train_last_n=15 --val_pool_keep_last_n_gens=20 --search_depth=8 --epochs_per_loop=1 --batch_size=256 --nnue_search_rate=0.0 --gen_data_threads=12