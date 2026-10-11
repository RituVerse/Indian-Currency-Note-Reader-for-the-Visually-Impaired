# Indian Currency Note Reader for the Visually Impaired

Entry for **BUILD WITH ML** – Coding Connoisseurs, FoET, University of Lucknow.

A web app that looks at an Indian banknote through the webcam (or an uploaded
photo), classifies it with a MobileNetV2 model running fully in the browser
(TensorFlow.js), and **speaks the value aloud** (e.g. "Five hundred rupees").

---

## Project Status

> Keep this section up to date on every push so anyone can continue from the repo alone.

### Done
- Step 0: repo set up, `.gitignore`, README skeleton.
- Step 1: dataset inspected (see "Dataset report" below).
- Step 2: `split_dataset.py` – de-duplicates (15 removed) and splits 180 unique
  images 70/15/15 with seed 42 → train 127 / val 27 / test 26.

### In progress
- Step 3: writing `train.py` (MobileNetV2 transfer learning).

### Remaining
1. Step 3: `train.py` – MobileNetV2 transfer learning + augmentation.
3. Step 4: evaluate on test set, confusion matrix image.
4. Step 5: export to TensorFlow.js into `/model`.
5. Step 6: `index.html`, `style.css`, `script.js` (webcam, upload, speech).
6. Step 7: local test of the page.
7. Step 8: full README.

### Decisions
- Model: MobileNetV2 (ImageNet weights, 224x224), transfer learning.
- Classes found in the ZIP: 10, 20, 50, 100, 200, 500, **2000** (the ZIP has a
  2000-rupee folder and **no `no_note` folder**).
- Confidence threshold for speaking: 85 %.
- Split: 70 / 15 / 15, random seed 42. Test set never used for training/tuning.

### Known issues / errors
- Dataset is small (195 images total) and contains 15 exact duplicate images,
  some shared between the original Train and Test folders. Duplicates are
  removed before splitting so the test set stays honest.

### How to reproduce
See "Reproduce" section below (filled in as steps complete).

### Files
| Path | Purpose |
|------|---------|
| `README.md` | this file |
| `.gitignore` | keeps dataset / large files out of git |
| `split_dataset.py` | pools Train+Test, removes duplicates, writes `dataset_split/{train,val,test}/<class>` |

---

## Dataset report (Step 1)

Source ZIP structure: `Train/` and `Test/` each with 7 class folders.

| Class folder | Train | Test | Total |
|---|---|---|---|
| Tennote (10) | 22 | 6 | 28 |
| Twentynote (20) | 22 | 6 | 28 |
| Fiftynote (50) | 22 | 6 | 28 |
| 1Hundrednote (100) | 22 | 6 | 28 |
| 2Hundrednote (200) | 22 | 6 | 28 |
| 5Hundrednote (500) | 22 | 6 | 28 |
| 2Thousandnote (2000) | 21 | 6 | 27 |
| **Total** | 153 | 42 | **195** |

- All 195 files are valid JPEGs (0 corrupt). 194 RGB, 1 RGBA.
- Class balance is good (27–28 per class). `no_note` class is **absent**.
- Image sizes vary widely (259x194 up to 1300x956) – all are resized to 224x224.
- 15 exact duplicate files were found (md5), including 2 Test images that also
  appear in Train. These are de-duplicated before our own split.
