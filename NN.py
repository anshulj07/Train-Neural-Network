import numpy as np
import tensorflow as tf


train_data = np.loadtxt('a2-train-data.txt')
train_labels = np.loadtxt('a2-train-label.txt')

test_data = np.loadtxt('a2-test-data.txt')

with open('a2-test-label.txt', 'r') as label_file:
    test_labels_raw = label_file.read()
test_labels_raw = test_labels_raw.replace('[', '').replace(']', '')
test_labels = np.array(test_labels_raw.split(','), dtype=float)


def create_model(num_hidden_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_hidden_units, activation='relu', input_shape=(1000,)),
        tf.keras.layers.Dense(1, activation='tanh')  # Output layer with tanh activation
    ])
    return model

def train_model(model, train_data, train_labels, num_epochs=10, learning_rate=0.001):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error')  
    
    model.fit(train_data, train_labels, epochs=num_epochs, verbose=2)

def evaluate_model(model, train_data, train_labels):
    predictions = model.predict(train_data)
    binary_predictions = [1 if pred >= 0 else -1 for pred in predictions]
    
   
    misclassifications = np.sum(binary_predictions != train_labels)
    return misclassifications


def predict(model, test_data):
    predictions = model.predict(test_data)
    binary_predictions = [1 if pred >= 0 else -1 for pred in predictions]
    return binary_predictions

if __name__ == "__main__":
    num_hidden_units = 64 
    model = create_model(num_hidden_units)
    train_model(model, train_data, train_labels)
    train_misclassifications = evaluate_model(model, train_data, train_labels)
    
    print(f'Training Misclassifications: {train_misclassifications}')

    predictions = predict(model, test_data)
    with open('predictions.txt', 'w') as f:
        f.write(' '.join(map(str, predictions)))

    with open('model.txt', 'w') as f:
        f.write(f'{num_hidden_units}\n')
        output_weights = model.layers[-1].get_weights()[0]
        f.write(' '.join(map(str, output_weights)) + '\n')
        for layer in model.layers[:-1]:
            hidden_weights = layer.get_weights()[0]
            f.write(' '.join(map(str, hidden_weights)) + '\n')
