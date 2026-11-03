import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
from datetime import datetime
import os
import pickle

# Read the training and testing files
print("Reading train_data.csv and test_data.csv...")
train_data = pd.read_csv('sum_cylinder.csv')
test_data = pd.read_csv('test_data_c.csv')

# Extract feature columns and target variables
print("Extracting features and target variables...")
X_train = train_data.drop(columns=['x', 'y'])
y_train = train_data[['x', 'y']]
X_test = test_data.drop(columns=['x', 'y'])
y_test = test_data[['x', 'y']]

# Define feature combination strategies
feature_combinations = ['V', 'P', 'all']
colors = ['blue', 'green', 'red']
labels = ['V features', 'P features', 'All features']

def create_model(input_shapes, is_fusion=False):

    if not is_fusion:
        # Single-input model
        inputs = Input(shape=(input_shapes,))
        x = inputs
    else:
        # Fusion model with two input branches
        input_V = Input(shape=(input_shapes[0],))
        input_P = Input(shape=(input_shapes[1],))
        x = concatenate([input_V, input_P])
        inputs = [input_V, input_P]
    
    # Shared hidden layers
    x = Dense(64, activation='relu')(x)
    x = Dense(48, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(2)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.002),
                 loss='mse',
                 metrics=['mae'])
    return model

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_save_dir = 'Cylinder_results'

save_dir = f'{base_save_dir}/run_{timestamp}'
os.makedirs(save_dir, exist_ok=True)
print("\nRunning single iteration")

predictions_list = []

for feature_combination, color, label in zip(feature_combinations, colors, labels):
    # Prepare data
    X_train_V = X_train.filter(regex='^V')
    X_train_P = X_train.filter(regex='^P')
    X_test_V = X_test.filter(regex='^V')
    X_test_P = X_test.filter(regex='^P')
    
    # Standardize the data
    scaler_V = StandardScaler()
    scaler_P = StandardScaler()
    X_train_V_scaled = scaler_V.fit_transform(X_train_V)
    X_train_P_scaled = scaler_P.fit_transform(X_train_P)
    X_test_V_scaled = scaler_V.transform(X_test_V)
    X_test_P_scaled = scaler_P.transform(X_test_P)
    
    model_configs = {
        'V': {
            'create_fn': lambda: create_model(X_train_V_scaled.shape[1]),
            'train_data': X_train_V_scaled,
            'test_data': X_test_V_scaled
        },
        'P': {
            'create_fn': lambda: create_model(X_train_P_scaled.shape[1]),
            'train_data': X_train_P_scaled,
            'test_data': X_test_P_scaled
        },
        'all': {
            'create_fn': lambda: create_model(
                (X_train_V_scaled.shape[1], X_train_P_scaled.shape[1]), 
                is_fusion=True
            ),
            'train_data': [X_train_V_scaled, X_train_P_scaled],
            'test_data': [X_test_V_scaled, X_test_P_scaled]
        }
    }
    
    config = model_configs[feature_combination]
    model = config['create_fn']()
    history = model.fit(config['train_data'], y_train,
                       epochs=300, batch_size=32, verbose=0)
    y_pred = model.predict(config['test_data'])
    
    # Save loss history
    loss_df = pd.DataFrame({
        'epoch': range(1, len(history.history['loss']) + 1),
        'loss': history.history['loss'],
        'mae': history.history['mae']
    })
    loss_df.to_csv(f'{save_dir}/{feature_combination}_loss_history.csv', index=False)
    
    predictions_list.append(y_pred)
    
    # Compute prediction errors
    errors = ((y_test.values - y_pred) ** 2).sum(axis=1)


# Save current run results
test_results = []
for feat, pred in zip(feature_combinations, predictions_list):
    # Compute evaluation metrics
    mse = ((y_test.values - pred) ** 2).mean(axis=0)
    rmse = np.sqrt(mse)
    mae = np.abs(y_test.values - pred).mean(axis=0)
    distances = np.sqrt(np.sum((y_test.values - pred) ** 2, axis=1))
    
    # Save prediction results
    result_df = pd.DataFrame({
        'run': 1,
        'feature': feat,
        'true_x': y_test['x'],
        'true_y': y_test['y'],
        'pred_x': pred[:, 0],
        'pred_y': pred[:, 1],
        'error_distance': distances
    })
    result_df.to_csv(f'{save_dir}/{feat}_predictions.csv', index=False)
    
    # Save the model
    model.save(f'{save_dir}/{feat}_model.h5')

# Save scalers using pickle
with open(f'{save_dir}/scaler_V.pkl', 'wb') as f:
    pickle.dump(scaler_V, f)
with open(f'{save_dir}/scaler_P.pkl', 'wb') as f:
    pickle.dump(scaler_P, f)
