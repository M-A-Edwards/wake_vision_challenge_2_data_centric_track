# Wake Vision Challenge 2 – Data-Centric Track Submission

**Participant:** Mark Edwards

---

##  Data Improvement Approach

###  Mislabeled Image Correction

**Method:**  
A custom `process_mislabeled_images` function utilizes YOLOv11x (`yolo11x.pt`) to detect persons (confidence > 0.4) in the `"non-person"` (class 0) directory of the training set. Images incorrectly labeled as non-person but detected to contain persons are automatically moved to the `"person"` (class 1) directory. All changes are logged in a timestamped log file for traceability.

An alternative `visualize_mislabeled_images` function previews up to 20 suspected mislabeled images, allowing manual inspection before applying corrections. This supports threshold tuning and helps make informed label adjustment decisions.

A CSV-based label correction pipeline can be easily integrated using the same log file.

**Purpose:**  
To correct label noise in the dataset, specifically reducing false negatives in the non-person class, leading to improved label accuracy and overall dataset quality.

---

## 🧪 Data Augmentation

**Method:**  
The `get_advanced_augmentation` function applies a set of soft augmentations:

-  Horizontal flip  
-  Rotation up to **15%**  
-  Zoom up to **35%**  
-  Translation up to **25%**  
-  Contrast adjustment up to **15%**  
-  Brightness adjustment up to **10%**

Experiments showed that aggressive settings for rotation, contrast, and brightness reduced accuracy, while softer transformations improved generalization. Although, increasing zoom and translation yielded significant accuracy improvements.

The `visualize_augmentation` function provides visual inspection of augmented samples.

**Purpose:**  
To increase dataset diversity and improve model robustness by exposing the model to realistic image variations.

---

##  Model Training

**Approach:**  
Due to computational limitations, training was performed in stages on progressively larger dataset subsets:

1. Initial training on `train-99`  
2. Expanded training on `train-99` + `train-11`  
3. Final training using `train-99`, `train-11`, and `train-12`

Each iteration benefited from the data correction and augmentation methods, resulting in a consistent accuracy gain of **3–4%**. The final model achieved an accuracy of **80.96%**.

