"""
split_dataset.py  -  Step 2
---------------------------
Takes the raw Kaggle ZIP folders (dataset_raw/Train/<class>, dataset_raw/Test/<class>),
pools ALL images per class, removes exact duplicate files, and writes a fresh
70 / 15 / 15 split into dataset_split/{train,val,test}/<class>/.

Why pool and re-split?
  The original Train/Test folders share some identical images, so the original
  Test set is "leaky". We make our own clean split with a fixed random seed so
  the result is reproducible and the test set is never seen in training.

Run:  python split_dataset.py
"""
import os
import random
import shutil
import hashlib

RAW_DIR = "dataset_raw"          # unzipped Kaggle archive
OUT_DIR = "dataset_split"        # output folder
SEED = 42                        # fixed seed -> same split every time
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remaining 0.15 = test

# Map the odd Kaggle folder names to simple class names used everywhere else.
CLASS_MAP = {
    "Tennote": "10",
    "Twentynote": "20",
    "Fiftynote": "50",
    "1Hundrednote": "100",
    "2Hundrednote": "200",
    "5Hundrednote": "500",
    "2Thousandnote": "2000",
}


def md5(path):
    """Hash a file so we can spot exact duplicates."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def collect_images():
    """Return {class_name: [unique image paths]} pooled from Train + Test."""
    images = {c: [] for c in CLASS_MAP.values()}
    seen_hashes = set()
    dup_count = 0
    for subset in ["Train", "Test"]:
        for raw_name, cls in CLASS_MAP.items():
            folder = os.path.join(RAW_DIR, subset, raw_name)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                path = os.path.join(folder, fname)
                h = md5(path)
                if h in seen_hashes:          # skip duplicates
                    dup_count += 1
                    continue
                seen_hashes.add(h)
                images[cls].append(path)
    print(f"Removed {dup_count} duplicate images.")
    return images


def main():
    random.seed(SEED)
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    images = collect_images()
    summary = []
    for cls, paths in images.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = round(n * TRAIN_FRAC)
        n_val = round(n * VAL_FRAC)
        parts = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train + n_val],
            "test": paths[n_train + n_val:],
        }
        for split, split_paths in parts.items():
            dest = os.path.join(OUT_DIR, split, cls)
            os.makedirs(dest, exist_ok=True)
            for i, p in enumerate(split_paths):
                # rename to avoid clashes between Train/Test files with same name
                shutil.copy(p, os.path.join(dest, f"{cls}_{i:03d}.jpg"))
        summary.append((cls, n, len(parts["train"]), len(parts["val"]), len(parts["test"])))

    print(f"\n{'class':>6} {'total':>6} {'train':>6} {'val':>5} {'test':>5}")
    for row in summary:
        print(f"{row[0]:>6} {row[1]:>6} {row[2]:>6} {row[3]:>5} {row[4]:>5}")
    tot = [sum(r[i] for r in summary) for i in range(1, 5)]
    print(f"{'ALL':>6} {tot[0]:>6} {tot[1]:>6} {tot[2]:>5} {tot[3]:>5}")


if __name__ == "__main__":
    main()
