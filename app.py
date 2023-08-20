import numpy as np
import logging
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from pyswarm import pso
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from pyswarm import pso
import types
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from tqdm import tqdm
# Keep a reference to the original pso function
original_pso = pso

def pso_with_logging(*args, **kwargs):
    # Call the original pso function
    xopt, fopt = original_pso(*args, **kwargs)

    # Log the result
    logging.info(f"PSO completed with best position {xopt} and best score {fopt}")

    return xopt, fopt

# Replace the original pso function with the new one
pso = pso_with_logging

# define callback for keras
class PSOProgressLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Finished epoch {epoch+1}")
        logging.info(f"Train loss: {logs['loss']}")
        logging.info(f"Train accuracy: {logs['accuracy']}")
        logging.info(f"Validation loss: {logs['val_loss']}")
        logging.info(f"Validation accuracy: {logs['val_accuracy']}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Logging is set up.")

# Generate Quantum Random Number for initial randomness
sampler = EmbeddingComposite(DWaveSampler())

def quantum_random_number(num_qubits=5):
    logging.info("Generating quantum random number...")
    response = sampler.sample_ising({i: 0.0 for i in range(num_qubits)}, {}, num_reads=1)
    most_common = next(iter(response)).values()
    logging.info("Quantum random number generated.")
    return sum([bit * 2**i for i, bit in enumerate(most_common)])

logging.info("Setting initial seed...")
initial_seed = quantum_random_number() % 2**32
np.random.seed(initial_seed)
logging.info("Initial seed set.")

# Data Preparation
logging.info("Reading and preparing data...")
with open('shakespeare.txt', 'r', encoding='utf-8') as file:
    shakespeare_text = file.read().lower().split('\n')

shakespeare_subset = shakespeare_text[:10000]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(shakespeare_subset)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in shakespeare_subset:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(seq) for seq in input_sequences])
padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
X, y = padded_sequences[:,:-1], padded_sequences[:,-1]
y = to_categorical(y, num_classes=total_words)

# Split the data into training and validation sets
logging.info("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=initial_seed)
logging.info("Data split into training and validation sets.")

# Tensegrity Weight Representation
class TensegrityWeights:
    def __init__(self, shape):
        logging.info("Initializing TensegrityWeights...")
        self.stiffness = np.random.rand(*shape) * 10
        self.shape = shape
        logging.info("TensegrityWeights initialized.")

    def get_normalized_weights(self):
        logging.info("Getting normalized weights...")
        return self.stiffness / 10

    def update_stiffness(self, weight_updates):
        logging.info("Updating stiffness...")
        adjustment = weight_updates * 10
        self.stiffness += adjustment
        self.stiffness = np.clip(self.stiffness, 0, 10)
        logging.info("Stiffness updated.")

def tensegrity_initializer(shape, dtype=None):
    logging.info("Initializing tensegrity...")
    tw = TensegrityWeights(shape)
    return tw.get_normalized_weights()

# Neural Network Model
def create_tensegrity_model(learning_rate=0.01, lstm_units=150, batch_size=16):
    logging.info("Creating tensegrity model...")
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1, embeddings_initializer=tensegrity_initializer))
    model.add(LSTM(lstm_units, return_sequences=True, kernel_initializer=tensegrity_initializer))
    model.add(LSTM(lstm_units, kernel_initializer=tensegrity_initializer))
    model.add(Dense(total_words, activation='softmax', kernel_initializer=tensegrity_initializer))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    logging.info("Tensegrity model created.")
    return model

# Quantum Annealing Integration
def discretize_weights(weights, num_bits=8):
    logging.info("Discretizing weights...")
    # Discretization is necessary because quantum annealers can only solve discrete optimization problems.
    # Reference: Lucas, Andrew. "Ising formulations of many NP problems." Frontiers in Physics 2 (2014): 5.
    min_weight = np.min(weights)
    max_weight = np.max(weights)
    step = (max_weight - min_weight) / (2**num_bits - 1)
    discretized_weights = np.round((weights - min_weight) / step)
    logging.info("Weights discretized.")
    return discretized_weights, min_weight, step

def continuous_weights(discretized_weights, min_weight, step):
    logging.info("Converting weights to continuous...")
    # After the quantum annealing process, we need to convert the weights back to continuous values.
    # Reference: Lucas, Andrew. "Ising formulations of many NP problems." Frontiers in Physics 2 (2014): 5.
    return discretized_weights * step + min_weight

sampler = LeapHybridSampler()

def create_qubo(weights, sample_size=1000):
    logging.info("Creating QUBO...")
    # Ensure sample_size is not larger than the size of weights
    sample_size = min(sample_size, len(weights))
    sample_indices = np.random.choice(len(weights), size=sample_size, replace=False)
    sampled_weights = weights[sample_indices]
    A = np.random.rand(sample_size, sample_size)
    b = np.random.rand(sample_size)
    Q = {}
    for i in range(sample_size):
        for j in range(sample_size):
            if i == j:
                Q[(i, j)] = b[i]
            else:
                Q[(i, j)] = A[i, j]
    logging.info("QUBO created.")
    return Q

def quantum_anneal_layer_weights(model, batch_size=1000, num_passes=5, patience=2, mutation_rate=0.1):
    logging.info("Starting quantum annealing...")
    # Quantum annealing is a global optimization method that uses quantum mechanics to find the minimum of a function.
    # Reference: Kadowaki, Tadashi, and Hidetoshi Nishimori. "Quantum annealing in the transverse Ising model." Physical Review E 58.5 (1998): 5355.
    start_time = time.time()
    total_weights = sum([np.prod(layer.get_weights()[0].shape) for layer in model.layers])
    processed_weights = 0
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()[0].flatten()
        best_weights = None
        best_fitness = float('inf')
        no_improvement_count = 0
        for pass_num in range(num_passes):
            logging.info(f"Starting pass {pass_num+1} of {num_passes}")
            for start in range(0, len(weights), batch_size):
                end = min(start + batch_size, len(weights))
                batch_weights = weights[start:end]
                logging.info(f"Processing batch {start//batch_size + 1} of {len(weights)//batch_size + 1} for layer {i + 1} of {len(model.layers)}")
                qubo = create_qubo(batch_weights)
                sampler = LeapHybridSampler()
                sampleset = sampler.sample_qubo(qubo)
                sample = sampleset.first.sample
                optimized_weights = np.array([sample[i] for i in range(len(batch_weights))])
                weights[start:end] = optimized_weights
                processed_weights += len(batch_weights)
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time * total_weights / processed_weights
                estimated_remaining_time = estimated_total_time - elapsed_time
                logging.info(f"Estimated remaining time: {estimated_remaining_time} seconds")
            fitness = objective_function(weights)
            if fitness < best_fitness:
                best_fitness = fitness
                best_weights = weights.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    logging.info(f"No improvement after {patience} passes, stopping early.")
                    break
                weights = best_weights.copy()  # revert to the best weights
            # Mutation
            for _ in range(int(mutation_rate * len(weights))):
                mutation_index = quantum_random_number() % len(weights)
                weights[mutation_index] = quantum_random_number() % 2**32
        weights = weights.reshape(layer.get_weights()[0].shape)
        layer.set_weights([weights])
    logging.info("Quantum annealing completed.")

# Optimization with PSO
call_count = 0
def objective_function(params):
    global call_count
    call_count += 1
    
    logging.info(f"Starting call {call_count} of objective function...")
    learning_rate = params[0]
    lstm_units = int(params[1])
    batch_size = int(params[2])
    model = create_tensegrity_model(learning_rate, lstm_units, batch_size)
    logging.info("Create Tensegrity model completed...")

    # Create a ModelCheckpoint callback
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    def lr_schedule(epoch):
        return learning_rate * (0.1 ** int(epoch / 10))

    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Pass the callback to the fit method
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=batch_size, verbose=2, callbacks=[PSOProgressLogger(), checkpoint, early_stopping, lr_scheduler])
    logging.info("model fit complete...")
    val_accuracy = history.history['val_accuracy'][-1]
    logging.info("Objective function completed.")
    return 1 - val_accuracy

lb = [0.001, 50, 10]  # lower bounds for learning rate, LSTM units, and batch size
ub = [0.1, 200, 32]  # upper bounds for learning rate, LSTM units, and batch size

# Log the start time of the PSO optimization
logging.info("Starting PSO optimization...")
start_time = time.time()

best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=50)

# Log the end time of the PSO optimization
end_time = time.time()
logging.info(f"PSO optimization completed in {end_time - start_time} seconds")

optimized_learning_rate = best_params[0]
model = create_tensegrity_model(optimized_learning_rate)

# Train the model
logging.info("Starting model training...")
start_time = time.time()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, verbose=1)
end_time = time.time()
logging.info(f"Model training completed in {end_time - start_time} seconds")

# Evaluate the model on the validation set
logging.info("Evaluating model...")
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
logging.info(f"Validation loss: {loss}")
logging.info(f"Validation accuracy: {accuracy}")

# Save the model
logging.info("Saving model...")
model.save('tensegrity_model.h5')
logging.info("Model saved as 'tensegrity_model.h5'")

logging.info("Starting quantum annealing for each layer...")
start_time = time.time()
quantum_anneal_layer_weights(model)
end_time = time.time()
logging.info(f"Quantum annealing completed in {end_time - start_time} seconds")

# Prediction function
def predict_next_word(model, tokenizer, text_sequence):
    logging.info("Predicting next word...")
    encoded_sequence = tokenizer.texts_to_sequences([text_sequence])[0]
    encoded_sequence = pad_sequences([encoded_sequence], maxlen=max_sequence_len-1, truncating='pre')
    predicted_probs = model.predict(encoded_sequence)[0]
    predicted_index = np.argmax(predicted_probs)
    logging.info("Next word predicted.")
    return tokenizer.index_word[predicted_index]

# Test the prediction
logging.info("Testing prediction...")
test_sequence = "To be or not to"
predicted_word = predict_next_word(model, tokenizer, test_sequence)
logging.info(f"Given the sequence '{test_sequence}', the predicted next word is: {predicted_word}")