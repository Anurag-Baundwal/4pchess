import argparse
import gc
import glob
import numpy as np
import os
import shutil
import subprocess
import tensorflow as tf
import time

# --- Binary Entry Format ---
ENTRY_DTYPE = np.dtype([
    ('board_state', np.uint8, (784,)),
    ('score', np.int16),
    ('player_turn', np.uint8),
    ('game_result', np.int8)
])

# --- Global Validation Pool ---
g_val_pool_boards = []
g_val_pool_scores = []

def generate_training_data(output_dir, search_depth, num_threads, num_positions, nnue_weights, nnue_rate):
    print(f'Generating {num_positions} positions into {output_dir}...')
    prog = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../bazel-bin/nnue/gen_data')
    if os.name == 'nt' and not prog.endswith('.exe'): prog += '.exe'

    args = [prog, output_dir, str(search_depth), str(num_threads), str(num_positions), str(nnue_rate)]
    if nnue_weights and os.path.exists(nnue_weights): args.append(nnue_weights)

    subprocess.run(args, check=True)

def load_binary_dir_to_numpy(data_dir):
    """Loads all .bin files in a directory into a single numpy array instantly."""
    bin_files = glob.glob(os.path.join(data_dir, 'data_*.bin'))
    chunks = []
    for f in bin_files:
        if os.path.getsize(f) > 0:
            chunks.append(np.fromfile(f, dtype=ENTRY_DTYPE))
    if not chunks:
        return np.array([], dtype=ENTRY_DTYPE)
    return np.concatenate(chunks)

def convert_scores_to_probs(scores):
    denom = -10.0 / tf.math.log(1.0 / 0.9 - 1.0)
    score = tf.cast(scores, tf.float32) / 100.0 / denom
    return tf.math.sigmoid(score)

@tf.keras.utils.register_keras_serializable()
def one_hot_layer_fn(board_tensor):
    one_hot = tf.one_hot(board_tensor, depth=7, axis=-1)
    batch_dim = tf.shape(one_hot)[0]
    return tf.reshape(one_hot, [batch_dim, 4, 14*14*7])

def create_dataset(boards_np, scores_np, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((boards_np, scores_np))
    
    def map_fn(board, score):
        board_reshaped = tf.reshape(board, [4, 14*14])
        probs = convert_scores_to_probs(score)
        return board_reshaped, tf.expand_dims(probs, axis=-1)

    dataset = dataset.shuffle(min(len(boards_np), 200000))
    return dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_model(train_data_dirs, val_boards, val_scores, last_model, save_dir, epochs, batch_size):
    print(f'Loading data into RAM from {len(train_data_dirs)} directories...')
    
    # Load all training data directly into RAM
    train_chunks = [load_binary_dir_to_numpy(d) for d in train_data_dirs]
    train_data = np.concatenate(train_chunks) if train_chunks else np.array([], dtype=ENTRY_DTYPE)
    
    if len(train_data) == 0:
        print("No training data found. Skipping.")
        return

    train_dataset = create_dataset(train_data['board_state'], train_data['score'], batch_size)
    val_dataset = create_dataset(val_boards, val_scores, batch_size) if len(val_boards) > 0 else None

    # Load or Create Model
    model = None
    if last_model and os.path.exists(last_model):
        try:
            model = tf.keras.models.load_model(last_model)
            print(f"Resumed model from {last_model}")
        except: pass

    if not model:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(4, 14*14), dtype=tf.int32),
            tf.keras.layers.Lambda(one_hot_layer_fn),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"Training on {len(train_data)} positions...")
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=1)

    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'nnue.h5'))

    # Save weights to CSV for C++ Eval
    dense_layers = [x for x in model.layers if isinstance(x, tf.keras.layers.Dense)]
    for i, layer in enumerate(dense_layers):
        weights = layer.get_weights()
        if len(weights) == 2:
            np.savetxt(os.path.join(save_dir, f'layer_{i}.kernel'), weights[0].flatten(), delimiter=',')
            np.savetxt(os.path.join(save_dir, f'layer_{i}.bias'), weights[1].flatten(), delimiter=',')

def train_rl_pipeline(args):
    output_dir = os.path.abspath(args.output_dir)
    model_dir = os.path.join(output_dir, 'models')
    archive_dir = os.path.join(output_dir, 'gen_models')
    os.makedirs(model_dir, exist_ok=True)

    gen_id = 1
    while os.path.exists(os.path.join(archive_dir, f'gen_{gen_id}')): gen_id += 1

    last_model = os.path.join(archive_dir, f'gen_{gen_id-1}', 'nnue.h5') if gen_id > 1 else None
    nnue_weights_dir = os.path.join(archive_dir, f'gen_{gen_id-1}') if gen_id > 1 else None

    global g_val_pool_boards, g_val_pool_scores

    while gen_id <= args.num_self_play_loops:
        print(f'\n========== Generation {gen_id} ==========')
        
        raw_train_dir = os.path.join(output_dir, f'raw_train_gen_{gen_id}')
        if os.path.exists(raw_train_dir): shutil.rmtree(raw_train_dir)
        
        # 1. Generate Data
        generate_training_data(raw_train_dir, args.search_depth, args.threads, args.positions, nnue_weights_dir, args.nnue_rate)
        
        # 2. Load and Split
        new_data = load_binary_dir_to_numpy(raw_train_dir)
        np.random.shuffle(new_data)
        
        val_split = int(len(new_data) * args.val_fraction)
        val_data = new_data[:val_split]
        train_data = new_data[val_split:]

        # Update Validation Pool
        g_val_pool_boards.extend(val_data['board_state'])
        g_val_pool_scores.extend(val_data['score'])
        max_val_size = int(args.val_fraction * args.positions * args.val_pool_gens)
        g_val_pool_boards = g_val_pool_boards[-max_val_size:]
        g_val_pool_scores = g_val_pool_scores[-max_val_size:]

        # Save Train Data permanently
        train_save_dir = os.path.join(output_dir, f'train_data_gen_{gen_id}')
        os.makedirs(train_save_dir, exist_ok=True)
        train_data.tofile(os.path.join(train_save_dir, 'data_0.bin'))
        shutil.rmtree(raw_train_dir)

        # 3. Train
        train_dirs = [os.path.join(output_dir, f'train_data_gen_{i}') for i in range(max(1, gen_id - args.train_last_n + 1), gen_id + 1)]
        
        train_model(
            train_dirs,
            np.array(g_val_pool_boards), np.array(g_val_pool_scores),
            last_model, model_dir, args.epochs, args.batch_size
        )

        # 4. Archive
        archive_path = os.path.join(archive_dir, f'gen_{gen_id}')
        shutil.copytree(model_dir, archive_path, dirs_exist_ok=True)
        
        last_model = os.path.join(archive_path, 'nnue.h5')
        nnue_weights_dir = archive_path
        gen_id += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--num_self_play_loops', default=100, type=int)
    parser.add_argument('--positions', default=100000, type=int)
    parser.add_argument('--val_fraction', default=0.1, type=float)
    parser.add_argument('--train_last_n', default=5, type=int)
    parser.add_argument('--val_pool_gens', default=10, type=int)
    parser.add_argument('--search_depth', default=6, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--nnue_rate', default=0.5, type=float)
    parser.add_argument('--threads', default=12, type=int)
    args = parser.parse_args()

    train_rl_pipeline(args)