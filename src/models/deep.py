import tensorflow as tf
from transformers import DistilBertTokenizer, TFAutoModelForSequenceClassification
from src.models.base_model import BaseModel
import numpy as np

class BERTClassifier(BaseModel):
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", max_len=128, name="bert"):
        super().__init__(name)
        self.max_len = max_len
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
        
        # Default compilation
        self._compile_model(lr=2e-5)

    def _compile_model(self, lr):
        """Helper to compile the model with a specific learning rate."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def freeze_backbone(self, num_layers_to_freeze=4):
        """Freezes the first N layers of the DistilBERT transformer."""
        # DistilBERT has 6 transformer layers
        for layer in self.model.layers[0].transformer.layer[:num_layers_to_freeze]:
            layer.trainable = False
        print(f"Froze the first {num_layers_to_freeze} layers of BERT.")

    def _tokenize(self, texts):
        return self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="tf"
        )

    def train(self, X_text, y, epochs=3, batch_size=8, validation_split=0.2, lr=None):
        """Modified train method to support learning rate adjustment."""
        if lr:
            self._compile_model(lr)
            
        tokenized_data = self._tokenize(X_text)
        y_array = np.array(y)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-7)
        ]

        self.model.fit(
            {"input_ids": tokenized_data["input_ids"], "attention_mask": tokenized_data["attention_mask"]},
            y_array,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

    def predict_proba(self, X_text):
        tokenized_data = self._tokenize(X_text)
        logits = self.model.predict(
            {"input_ids": tokenized_data["input_ids"], "attention_mask": tokenized_data["attention_mask"]},
            verbose=0
        ).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()
        return probs[:, 1]