import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import log_loss, accuracy_score

class DeepModel:
    def __init__(self, X_train_tensor, y_train, X_val_tensor, y_val):
        self.X_train = X_train_tensor
        self.y_train = y_train
        self.X_val = X_val_tensor
        self.y_val = y_val
        
        self.timesteps = self.X_train.shape[1]
        self.features = self.X_train.shape[2]
        
        self.build_model()
        self.train()

    def build_model(self):
        inputs = layers.Input(shape=(self.timesteps, self.features))

        x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding='same')(inputs)

        attention_output = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)

        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        x = layers.GlobalAveragePooling1D()(x) 
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(16, activation='relu')(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )
        
        self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=30,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

    def get_performance(self):
        predictions = self.get_predictions()
        return {
            'LogLoss': log_loss(self.y_val, predictions), 
            'Acc': accuracy_score(self.y_val, predictions > 0.5)
        }

    def get_predictions(self):
        return self.model.predict(self.X_val, verbose=0).flatten()