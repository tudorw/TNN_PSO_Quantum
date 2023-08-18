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
def quantum_random_number(num_qubits=5):
    logging.info("Generating quantum random number...")
    sampler = EmbeddingComposite(DWaveSampler())
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
def create_tensegrity_model(learning_rate=0.01):
    logging.info("Creating tensegrity model...")
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1, embeddings_initializer=tensegrity_initializer))
    model.add(LSTM(150, return_sequences=True, kernel_initializer=tensegrity_initializer))
    model.add(LSTM(100, kernel_initializer=tensegrity_initializer))
    model.add(Dense(total_words, activation='softmax', kernel_initializer=tensegrity_initializer))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    logging.info("Tensegrity model created.")
    return model

# Quantum Annealing Integration
def discretize_weights(weights, num_bits=8):
    logging.info("Discretizing weights...")
    min_weight = np.min(weights)
    max_weight = np.max(weights)
    step = (max_weight - min_weight) / (2**num_bits - 1)
    discretized_weights = np.round((weights - min_weight) / step)
    logging.info("Weights discretized.")
    return discretized_weights, min_weight, step

def continuous_weights(discretized_weights, min_weight, step):
    logging.info("Converting weights to continuous...")
    return discretized_weights * step + min_weight

def create_qubo(weights):
    logging.info("Creating QUBO...")
    A = np.random.rand(len(weights), len(weights))
    b = np.random.rand(len(weights))
    Q = {}
    for i in range(len(weights)):
        for j in range(len(weights)):
            if i == j:
                Q[(i, j)] = b[i]
            else:
                Q[(i, j)] = A[i, j]
    logging.info("QUBO created.")
    return Q

# def quantum_anneal(weights):
#     qubo = create_qubo(weights)
#     sampler = LeapHybridSampler()
#     sampleset = sampler.sample_qubo(qubo)
#     sample = sampleset.first.sample
#     optimized_weights = np.array([sample[i] for i in range(len(weights))])
#     return optimized_weights

# def quantum_anneal(weights):
#     logging.info("Starting quantum annealing...")
#     qubo = create_qubo(weights)
#     sampler = EmbeddingComposite(DWaveSampler())
#     sampleset = sampler.sample_qubo(qubo, num_reads=1000)
#     sample = sampleset.first.sample
#     optimized_weights = np.array([sample[i] for i in range(len(weights))])
#     logging.info("Quantum annealing completed.")
#     return optimized_weights

def quantum_anneal_layer_weights(model):
    logging.info("Starting quantum annealing...")
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()[0].flatten()
        qubo = create_qubo(weights)
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample_qubo(qubo, num_reads=1000)
        sample = sampleset.first.sample
        optimized_weights = np.array([sample[i] for i in range(len(weights))])
        optimized_weights = optimized_weights.reshape(layer.get_weights()[0].shape)
        layer.set_weights([optimized_weights])
    logging.info("Quantum annealing completed.")

# Optimization with PSO
call_count = 0
def objective_function(params):
    global call_count
    call_count += 1
    
    logging.info(f"Starting call {call_count} of objective function...")
    learning_rate = params[0]
    model = create_tensegrity_model(learning_rate)
    logging.info("Create Tensegrity model completed...")

    # Create a ModelCheckpoint callback
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Pass the callback to the fit method
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=16, verbose=2, callbacks=[PSOProgressLogger(), checkpoint])
    logging.info("model fit complete...")
    val_accuracy = history.history['val_accuracy'][-1]
    logging.info("Objective function completed.")
    return 1 - val_accuracy

lb = [0.001]
ub = [0.1]

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

# # Quantum Annealing for Embedding Layer
# logging.info("Starting quantum annealing for embedding layer...")
# start_time = time.time()
# embedding_weights = model.layers[0].get_weights()[0].flatten()
# discretized_weights, min_weight, step = discretize_weights(embedding_weights)
# optimized_discretized_weights = quantum_anneal(discretized_weights)
# optimized_weights = continuous_weights(optimized_discretized_weights, min_weight, step)
# optimized_weights = optimized_weights.reshape(model.layers[0].get_weights()[0].shape)
# model.layers[0].set_weights([optimized_weights])
# end_time = time.time()
# logging.info(f"Quantum annealing completed in {end_time - start_time} seconds")
# Quantum Annealing for Each Layer

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