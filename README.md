```markdown
# Fashion MNIST Multi-Class Classification with Feedforward Neural Network (PyTorch)

This project implements a **feedforward neural network** using **PyTorch** for multi-class classification on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. The model is trained and evaluated under different configurations to analyze the impact of varying the number of hidden layers and training epochs.

---

## 📋 **Objective**

- **Design, implement, and evaluate** a feedforward neural network for multi-class image classification.
- **Experiment** with different network depths (hidden layers) and training durations (epochs).
- **Analyze** how these variations affect model performance in terms of accuracy and loss[2].

---

## 🛠️ **Implementation Details**

- **Framework:** PyTorch
- **Model:** Feedforward Neural Network (Fully Connected)
- **Dataset:** Fashion MNIST (10 classes of fashion items)
- **Task:** Multi-class classification
- **Evaluation Metrics:** Training loss, training accuracy, per-class accuracy, confusion matrix, training time

---

## 🗂️ **Project Structure**

- `ann-mnist-dataseta.ipynb`: Main Jupyter notebook containing all code for data loading, model definition, training, evaluation, and visualization[1].
- (Optional) Additional notebooks for different configurations (see below).

---

## 🧑‍💻 **How It Works**

1. **Data Loading & Preprocessing**
   - Downloads and normalizes the Fashion MNIST dataset.
   - Splits into training and test sets.
2. **Model Architecture**
   - Configurable feedforward neural network.
   - Typical configuration:  
     - Input: 784 (28x28 images flattened)
     - 1–3 hidden layers (e.g., 128 units each, with ReLU activation)
     - Output: 10 classes (softmax)
3. **Training**
   - Cross-entropy loss
   - Adam optimizer
   - Configurable batch size, learning rate, and epochs
4. **Evaluation**
   - Reports training loss and accuracy per epoch.
   - Evaluates overall and per-class accuracy on the test set.
   - Visualizes confusion matrix and sample predictions.
5. **Visualization**
   - Plots training loss and accuracy curves.
   - Displays confusion matrix and sample images with predicted labels[1].

---

## ⚙️ **Configuration Experiments**

You can experiment with:
- **Number of Hidden Layers:** 1, 2, or 3
- **Number of Epochs:** 5 or 10

**Key Observations:**
- More epochs (10 vs 5) consistently improve accuracy.
- Adding more hidden layers (2 or 3) gives only a small improvement.
- Best result: **91.17% accuracy** with 2 hidden layers and 10 epochs[2].

---

## 🚀 **Quick Start**

1. **Clone this repository** and install dependencies:
   ```
   pip install torch torchvision matplotlib numpy
   ```
2. **Run the notebook**:
   - Open `ann-mnist-dataseta.ipynb` in Jupyter and run all cells.
   - Adjust hyperparameters at the top of the notebook as desired.

---

## 📊 **Results Summary**

| Hidden Layers | Epochs | Best Accuracy (%) |
|:-------------:|:------:|:----------------:|
|      1        |   5    |      ~88         |
|      1        |  10    |      ~90         |
|      2        |   5    |      ~89         |
|      2        |  10    |    **91.17**     |
|      3        |   5    |      ~89         |
|      3        |  10    |      ~91         |

- **More epochs** help the model learn better.
- **More hidden layers** give only minor improvements beyond one or two[2].

---

## 📦 **Dependencies**

- Python 3.8+
- torch
- torchvision
- numpy
- matplotlib

---

## 👤 **Author**

Muhammad Shakaib Arsalan 

---

