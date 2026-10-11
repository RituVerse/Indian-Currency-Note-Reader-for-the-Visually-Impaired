"""
train.py  -  Step 3 + Step 4
----------------------------
Trains a MobileNetV2 transfer-learning classifier on Indian currency notes,
then evaluates it ONCE on the held-out test set and saves a confusion matrix.

Pipeline
  1. Load dataset_split/{train,val,test} (made by split_dataset.py).
  2. Data augmentation on the TRAIN set only: rotation, zoom, brightness,
     contrast, small shifts.  (No horizontal flip: banknotes have text and
     portraits, a mirrored note is not a realistic input.)
  3. MobileNetV2 (ImageNet weights, frozen) + GlobalAveragePooling + Dropout +
     Dense softmax head.   Phase 1: train the head only.
  4. Phase 2 (fine-tuning): unfreeze the top ~30 layers of MobileNetV2 and
     train with a very small learning rate.
  5. Evaluate on the TEST set (never used before this point), print accuracy
     and classification report, save confusion_matrix.png.
  6. Save the Keras model to saved_model/currency_model.keras for TF.js export.

Run:  python train.py
"""
import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")                      # no display in the sandbox
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------------------------------------------- settings --
DATA_DIR = "dataset_split"
IMG_SIZE = (224, 224)                     # MobileNetV2 native input
BATCH = 16
SEED = 42
EPOCHS_HEAD = 15                           # phase 1
EPOCHS_FINE = 15                           # phase 2
OUT_MODEL = "saved_model/currency_model.keras"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------------------- load data ----
def load_split(name, shuffle):
    """Return a tf.data.Dataset of (image, label) from dataset_split/<name>."""
    return keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, name),
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=shuffle,
        seed=SEED,
        label_mode="int",
    )


train_ds = load_split("train", shuffle=True)
val_ds = load_split("val", shuffle=False)
test_ds = load_split("test", shuffle=False)
class_names = train_ds.class_names          # e.g. ['10','100','20','200',...]
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# --------------------------------------------------------- augmentation -----
# Applied inside the model, so it is only active during training (model.fit).
augment = keras.Sequential(
    [
        layers.RandomRotation(0.08),                 # +-~30 degrees
        layers.RandomZoom(0.15, 0.15),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomBrightness(0.25),               # lighting changes
        layers.RandomContrast(0.25),
    ],
    name="augmentation",
)

# ---------------------------------------------------------------- model -----
base = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False                            # phase 1: frozen backbone

inputs = keras.Input(shape=IMG_SIZE + (3,), name="image")
x = augment(inputs)
# MobileNetV2 expects pixels scaled to [-1, 1]. Doing it inside the model means
# the browser just feeds raw 0-255 pixels. Same math as
# keras.applications.mobilenet_v2.preprocess_input, written with plain layers
# so the TF.js converter has nothing exotic to translate.
x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="to_minus1_1")(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax", name="note")(x)
model = keras.Model(inputs, outputs, name="currency_mobilenetv2")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary(show_trainable=False)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True
    )
]

# ------------------------------------------------------- phase 1: head ------
print("\n=== Phase 1: training classifier head ===")
hist1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks)

# ------------------------------------------------- phase 2: fine-tuning -----
print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 ===")
base.trainable = True
for layer in base.layers[:-30]:           # keep all but the last 30 layers frozen
    layer.trainable = False
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),   # tiny LR so we don't wreck weights
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
hist2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=callbacks)

# ------------------------------------------------- training curves plot -----
acc = hist1.history["accuracy"] + hist2.history["accuracy"]
val_acc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
plt.figure(figsize=(7, 4))
plt.plot(acc, label="train accuracy")
plt.plot(val_acc, label="val accuracy")
plt.axvline(len(hist1.history["accuracy"]) - 0.5, color="grey", ls="--", label="start fine-tuning")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Training curves")
plt.tight_layout(); plt.savefig("training_curves.png", dpi=120); plt.close()

# ---------------------------------------------- Step 4: TEST evaluation -----
print("\n=== Evaluating on the TEST set (first and only time) ===")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"TEST accuracy: {test_acc:.4f}   TEST loss: {test_loss:.4f}")

y_true, y_pred = [], []
for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    y_true.extend(labels.numpy()); y_pred.extend(np.argmax(probs, axis=1))
y_true, y_pred = np.array(y_true), np.array(y_pred)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(len(class_names)), [f"Rs {c}" for c in class_names], rotation=45)
plt.yticks(range(len(class_names)), [f"Rs {c}" for c in class_names])
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Confusion matrix - test accuracy {test_acc*100:.1f}%")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120); plt.close()

# Which notes get confused?
confused = [(class_names[i], class_names[j], int(cm[i, j]))
            for i in range(len(class_names)) for j in range(len(class_names))
            if i != j and cm[i, j] > 0]
print("Confusions (true -> predicted, count):", confused or "none")

# ------------------------------------------------------------- save ---------
os.makedirs("saved_model", exist_ok=True)
os.makedirs("model", exist_ok=True)
model.save(OUT_MODEL)
# The browser needs to know which output index is which note.
with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)
with open("test_results.json", "w") as f:
    json.dump({"test_accuracy": float(test_acc), "test_loss": float(test_loss),
               "confusion_matrix": cm.tolist(), "class_names": class_names,
               "confusions": confused,
               "epochs_run": len(acc)}, f, indent=2)
print(f"\nSaved model to {OUT_MODEL}, results to test_results.json")
