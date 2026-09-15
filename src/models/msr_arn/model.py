import os
import json
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, Reshape, GlobalAveragePooling1D,
    LayerNormalization, Concatenate, GlobalMaxPooling1D,
    Multiply, Add, Activation, Input, MultiHeadAttention, Lambda
)
from tensorflow.keras.optimizers import Adam

def create_msr_arn_model(input_dim, learning_rate=1e-3):
    """
    Constructs the Multi-Scale Residual Attentive Regression Network (MSR-ARN).
    """
    # Helper: GELU activation wrapper
    gelu = lambda x: Activation(tf.nn.gelu)(x)

    def MultiScaleResidual(in_tensor, filters):
        # Parallel kernels capture different temporal scales
        p1 = Conv1D(filters=filters, kernel_size=3, padding='same')(in_tensor)
        p1 = gelu(p1)
        p2 = Conv1D(filters=filters, kernel_size=7, padding='same')(in_tensor)
        p2 = gelu(p2)
        p3 = Conv1D(filters=filters, kernel_size=15, padding='same')(in_tensor)
        p3 = gelu(p3)

        merged = Concatenate(axis=-1)([p1, p2, p3])
        merged = Conv1D(filters=filters, kernel_size=1, padding='same')(merged)
        merged = gelu(merged)

        # Squeeze-and-Excitation (SE) block
        se = GlobalAveragePooling1D()(merged)
        se = Dense(max(4, filters // 8), activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        se = Reshape((1, filters))(se)
        se_scaled = Multiply()([merged, se])

        # Shortcut connection
        in_channels = int(in_tensor.shape[-1])
        if in_channels != filters:
            shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(in_tensor)
        else:
            shortcut = in_tensor

        out = Add()([se_scaled, shortcut])
        out = LayerNormalization()(out)
        return out

    def DilatedGatedConv(in_tensor, filters, dilation_rate=1):
        conv_f = Conv1D(filters=filters, kernel_size=3, dilation_rate=dilation_rate, padding='same')(in_tensor)
        conv_g = Conv1D(filters=filters, kernel_size=3, dilation_rate=dilation_rate, padding='same')(in_tensor)
        gated = Multiply()([Activation('tanh')(conv_f), Activation('sigmoid')(conv_g)])
        gated = Conv1D(filters=filters, kernel_size=1, padding='same')(gated)

        in_channels = int(in_tensor.shape[-1])
        if in_channels != filters:
            proj = Conv1D(filters=filters, kernel_size=1, padding='same')(in_tensor)
        else:
            proj = in_tensor

        out = Add()([gated, proj])
        out = LayerNormalization()(out)
        return out

    def ResidualDenseBlock(inp, units, dropout_rate=0.1):
        inp_dim = int(inp.shape[-1])
        if inp_dim != units:
            shortcut = Dense(units)(inp)
        else:
            shortcut = inp
        
        h = Dense(units, activation=tf.nn.gelu)(inp)
        h = LayerNormalization()(h)
        h = Dropout(dropout_rate)(h)
        h = Dense(units, activation=tf.nn.gelu)(h)
        
        out = Add()([shortcut, h])
        out = LayerNormalization()(out)
        return out

    # --- Model Definition ---
    inp = Input(shape=(input_dim,), name='input_vector')
    
    # 1. Reshape flat vector to sequence
    x = Reshape((input_dim, 1), name='reshape_to_seq')(inp)

    # 2. Convolutional Stem
    stem = Conv1D(filters=64, kernel_size=3, padding='same')(x)
    stem = gelu(stem)
    stem = LayerNormalization()(stem)
    stem = Dropout(0.05)(stem)

    # 3. Multi-Scale Residual Modules
    m1 = MultiScaleResidual(stem, filters=128)
    m2 = MultiScaleResidual(m1, filters=128)

    # 4. Dilated Gated Convolutional Stack
    d1 = DilatedGatedConv(m2, filters=128, dilation_rate=1)
    d2 = DilatedGatedConv(d1, filters=128, dilation_rate=2)
    d3 = DilatedGatedConv(d2, filters=128, dilation_rate=4)

    # 5. Temporal Self-Attention (Transformer Block)
    attn_input = LayerNormalization()(d3)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(attn_input, attn_input)
    attn_out = Dropout(0.1)(attn_out)
    attn_out = Add()([attn_out, attn_input])
    attn_out = LayerNormalization()(attn_out)

    ff = Dense(256, activation=tf.nn.gelu)(attn_out)
    ff = Dense(128)(ff)
    ff = Dropout(0.1)(ff)
    attn_out = Add()([attn_out, ff])
    attn_out = LayerNormalization()(attn_out)

    # 6. Global Context Aggregation
    gavg = GlobalAveragePooling1D()(attn_out)
    gmax = GlobalMaxPooling1D()(attn_out)
    gstd = Lambda(lambda t: tf.math.reduce_std(t + 1e-6, axis=1))(attn_out)
    global_feat = Concatenate()([gavg, gmax, gstd])

    # 7. FiLM Modulation
    film_scale = Dense(units=global_feat.shape[-1], activation='sigmoid')(global_feat)
    film_shift = Dense(units=global_feat.shape[-1], activation='tanh')(global_feat)
    modulated = Multiply()([global_feat, film_scale])
    modulated = Add()([modulated, film_shift])

    # 8. Regression Head
    h = Dense(512, activation=tf.nn.gelu)(modulated)
    h = ResidualDenseBlock(h, units=512, dropout_rate=0.2)
    h = ResidualDenseBlock(h, units=512, dropout_rate=0.15)
    
    h = Dense(256, activation=tf.nn.gelu)(h)
    h = Dropout(0.1)(h)
    h = Dense(64, activation=tf.nn.gelu)(h)

    out = Dense(1, activation='linear', name='pIC50_out')(h)

    model = Model(inputs=inp, outputs=out, name='MSR_ARN_regressor')
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='mse', 
                  metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    return model


if __name__ == "__main__":
    # ==============================================================================
    # NVIDIA DGX A100 STATION PARALLEL PROCESSING SETUP
    # ==============================================================================
    
    # MirroredStrategy will symmetrically distribute computation across all available A100 GPUs.
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of GPUs being used for DGX A100: {strategy.num_replicas_in_sync}")

    # Core File Paths required for scaling, preprocessing, and model init
    feature_columns_path = "feature_columns.pkl"
    feature_scaler_path = "feature_scaler.pkl"
    model_meta_path = "model_meta.json"
    dataset_path = "MMP_12_dataset.sdf"
    pretrained_weights_path = "msr_arn.weights.h5" 
    
    print(f"\nDataset allocated at: {dataset_path}")

    # Load processing artifacts if they are present
    input_dim = None

    if os.path.exists(feature_columns_path):
        feature_columns = joblib.load(feature_columns_path)
        input_dim = len(feature_columns)
        print(f"Successfully loaded feature columns. Detected input dimension: {input_dim}")
    else:
        print("Warning: feature_columns.pkl is missing. Falling back to an arbitrary input_dim=200")
        input_dim = 200

    if os.path.exists(model_meta_path):
        with open(model_meta_path, 'r') as f:
            model_meta = json.load(f)
            print(f"Successfully loaded Model Meta settings: {model_meta}")
            
    if os.path.exists(feature_scaler_path):
        scaler = joblib.load(feature_scaler_path)
        print("Successfully loaded feature scaler.")

    # ==============================================================================
    # MODEL INITIALIZATION (Distributed Context)
    # ==============================================================================
    
    # Initialize the entire model architecture inside the MirroredStrategy scope
    # This ensures variables are created uniformly across multiple DGX A100s memory
    with strategy.scope():
        # Scale the learning rate linearly based on the number of workers/GPUs.
        base_learning_rate = 1e-3
        scaled_learning_rate = base_learning_rate * strategy.num_replicas_in_sync

        print("\nConstructing MSR-ARN Model inside Strategy Scope...")
        model = create_msr_arn_model(input_dim=input_dim, learning_rate=scaled_learning_rate)
        
        # Load pre-trained weights parallelized across the cluster if available
        if os.path.exists(pretrained_weights_path):
            print(f"Loading pre-trained weights from {pretrained_weights_path}...")
            model.load_weights(pretrained_weights_path)
            print("Pre-trained DGX scale weights loaded successfully.")
        else:
            print(f"Notice: Pre-trained weights file not found at {pretrained_weights_path}. Initialized randomly.")
            
    print("\nDGX A100 Model Initialization & Setup Complete!")
