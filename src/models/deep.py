# src/models/deep.py
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # ensure compatibility with transformers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import tf_keras as tfk   # use separate keras package if needed
from src.models.base_model import BaseModel


class BERTClassifier(BaseModel):
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english",
                 max_len=128, name="bert", from_pt=True):
        super().__init__(name)

        self.max_len = max_len
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Mixed precision
        tfk.mixed_precision.set_global_policy("mixed_float16")

        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, from_pt=from_pt
        )
        self._compile_model(lr=2e-5)

    def save(self, path: str):
        """Save pretrained model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, name="bert_loaded"):
        """Load from saved directory."""
        return cls(model_name=path, name=name, from_pt=False)

    def _compile_model(self, lr):
        self.model.compile(
            optimizer=tfk.mixed_precision.LossScaleOptimizer(
                tfk.optimizers.Adam(learning_rate=lr)
            ),
            loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def freeze_backbone(self, num_layers_to_freeze=4):
        """Freeze first N transformer layers."""
        for layer in self.model.layers[0].transformer.layer[:num_layers_to_freeze]:
            layer.trainable = False

# Inside src/models/deep.py, update your train() definition:

    def train(self, X_text, y, epochs=3, batch_size=16, validation_split=0.2, lr=None, patience=3):
        if lr:
            self._compile_model(lr)

        X_train, X_val, y_train, y_val = train_test_split(
            np.array(X_text), np.array(y),
            test_size=validation_split, stratify=y, random_state=42
        )

        train_enc = self.tokenizer(list(X_train), truncation=True, max_length=self.max_len, padding=False)
        val_enc = self.tokenizer(list(X_val), truncation=True, max_length=self.max_len, padding=False)

        def make_gen(enc, labels):
            def gen():
                for i in range(len(labels)):
                    yield ({"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}, labels[i])
            return gen

        output_signature = (
            {"input_ids": tf.TensorSpec((None,), tf.int32), "attention_mask": tf.TensorSpec((None,), tf.int32)},
            tf.TensorSpec((), tf.int32),
        )

        train_ds = tf.data.Dataset.from_generator(make_gen(train_enc, y_train), output_signature=output_signature)
        val_ds = tf.data.Dataset.from_generator(make_gen(val_enc, y_val), output_signature=output_signature)

        train_ds = train_ds.shuffle(buffer_size=min(len(X_train), 1000)).padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # --- DYNAMIC CALLBACKS ---
        callbacks = [
            tfk.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=patience,               # Uses the argument
                restore_best_weights=True
            ),
            tfk.callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5,                      # Gentler drop 
                patience=max(1, patience - 1),   # Drop LR just before stopping
                min_lr=1e-7
            ),
        ]

        return self.model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1
        )

    def predict_proba(self, X_text):
        tok = self.tokenizer(
            list(X_text), padding=True, truncation=True,
            max_length=self.max_len, return_tensors="tf"
        )
        logits = self.model.predict(
            {
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
            },
            verbose=0,
        ).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()
        return probs[:, 1]