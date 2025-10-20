#  Logistic Regression as a Single Neuron

This project demonstrates how **Logistic Regression** can be viewed as the simplest form of a **neural network** — a single neuron that takes inputs, applies weights and a bias, passes them through a **sigmoid activation**, and outputs a probability.

---

##  Project Overview
Logistic Regression is often taught as a statistical model, but it’s also the mathematical foundation of neural networks.  
In this project, we:
- Build and visualize a **binary classification** problem using `scikit-learn`.
- Train a **Logistic Regression** model to predict class labels.
- Plot the **decision boundary** and **confusion matrix**.
- Explain how the sigmoid function transforms linear combinations into probabilities.

---

##  Key Concepts
| Concept | Explanation |
|----------|--------------|
| **Sigmoid Function** | Squashes any real number into a range between 0 and 1, representing probability. |
| **Weights & Bias** | Determine the slope and position of the decision boundary. |
| **Decision Boundary** | The line that separates the two predicted classes. |
| **Threshold (0.5)** | Probabilities above → class 1, below → class 0. |

---
##  Visualization & Results

### Figure 1 — Dataset Overview  
Shows the structure of the dataset with columns like **User ID**, **Gender**, **Age**, **EstimatedSalary**, and **Purchased**.  
This helps verify data types, missing values, and memory usage before training the model.  
![Figure 1](https://github.com/pratik001010/Visualizing-Logistic-Regression-and-Neural-Networks/blob/b864161135e833276901744510ae04eed4423be5/figure1.png)

---

###  Figure 2 — Model Performance Report  
Displays the **classification report** with precision, recall, F1-score, and overall test accuracy of **84%**, showing that the model performs reliably on unseen data.  
Also includes the **confusion matrix** illustrating correctly and incorrectly classified samples.  
![Figure 2](https://github.com/pratik001010/Visualizing-Logistic-Regression-and-Neural-Networks/blob/b864161135e833276901744510ae04eed4423be5/figuire2.png)

---

###  Figure 3 — Decision Boundary Visualization  
Depicts the **decision regions** formed by the logistic regression model.  
Here, **Age** and **Estimated Salary** are used as predictors to separate the classes (purchased vs not purchased).  
The shaded areas represent predicted zones for each class.  
![Figure 3](https://github.com/pratik001010/Visualizing-Logistic-Regression-and-Neural-Networks/blob/b864161135e833276901744510ae04eed4423be5/figure%203.png)

---
###  Figure 4 — ROC Curve (AUC = 0.91)  
The **Receiver Operating Characteristic (ROC)** curve shows how well the model distinguishes between positive and negative classes.  
An **AUC of 0.91** indicates excellent classification capability and strong model performance.  
![Figure 4](https://github.com/pratik001010/Visualizing-Logistic-Regression-and-Neural-Networks/blob/b864161135e833276901744510ae04eed4423be5/figure%204.png)

---

###  Figure 5 — Dataset Sample (Head)  
Shows the first few rows of the dataset after preprocessing.  
It provides a quick look at the feature values used by the logistic regression model for learning.  
![Figure 5](https://github.com/pratik001010/Visualizing-Logistic-Regression-and-Neural-Networks/blob/b864161135e833276901744510ae04eed4423be5/figure%205.png)


## 🐍 Implementation Steps
1. Generate a dataset using `make_classification()`  
2. Visualize the points with `matplotlib`  
3. Train a `LogisticRegression()` model  
4. Evaluate accuracy using a confusion matrix  
5. Plot the decision boundary showing model predictions  

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create synthetic data
X, y = make_classification(n_samples=200, n_features=2,
                           n_redundant=0, n_informative=2,
                           random_state=42, n_clusters_per_class=1)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = LogisticRegression()
model.fit(X_train, y_train)


---
