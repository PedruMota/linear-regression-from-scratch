# 📉 Linear Regression from Scratch using Gradient Descent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

> **A vectorized, modular implementation of Linear Regression using Gradient Descent, benchmarked against Scikit-Learn in high-dimensional biological scenarios.**

---

## Project Overview

While `sklearn.linear_model.LinearRegression` is the industry standard, its reliance on the **Singular Value Decomposition (SVD)** analytical solution creates significant memory bottlenecks when handling high-dimensional datasets.

To address this, this project explores the use of **Gradient Descent** as an iterative alternative. Instead of attempting to solve the entire system at once through computationally expensive matrix decompositions, Gradient Descent approaches the optimal solution in steps. This study aims to demonstrate how this iterative process can be more resource-efficient in high-dimensional contexts, providing a practical way to manage memory usage without sacrificing the model's ability to learn from the data.

### Key Features
* **Vectorized Implementation:** Fully optimized matrix operations using NumPy (no slow `for` loops).
* **Automatic Feature Engineering:** Built-in Z-Score Normalization and One-Hot Encoding handling.
* **Dual-Scenario Validation:** Benchmarked using generated **Synthetic Data** across two distinct domains:
    * **Agro-Ecological Case:** Verified convergence and prediction accuracy against Scikit-Learn.
    * **Single-Cell Genomics Case:** Stress-tested memory efficiency on high-dimensional data ($N=5,000, P=20,000$).
* **Scikit-Learn Compatible:** Follows the standard `.fit()` and `.predict()` API design.

---

## Theoretical Foundation & Algorithm

### Scikit-Learn Approach
While the textbook definition of Linear Regression uses the **Normal Equation** $\theta = (X^T X)^{-1} X^T y$, production-grade libraries like `scikit-learn` avoid this method due to numerical instability when features are correlated.

Instead, `sklearn.linear_model.LinearRegression` relies on **Singular Value Decomposition (SVD)** of the data matrix $X$ (specifically using LAPACK's `gelsd` driver). It decomposes the matrix such that:
$$X = U \Sigma V^T$$
The solver then computes the pseudo-inverse to find the weights. While numerically stable, this approach requires decomposing the entire matrix in memory, leading to a computational cost of roughly **$O(N \cdot P^2)$** or **$O(P^3)$**. In our Genomics scenario ($P=20,000$), this matrix decomposition becomes the primary memory bottleneck.

### Iterative Approach
To bypass the memory cost of matrix decomposition, I have implemented **Batch Gradient Descent**. Instead of solving the system in a single step, we optimize the weights iteratively by following the slope of the error surface.



**Algorithm Steps:**
1.  **Prediction:** $\hat{y} = Xw + b$
2.  **Cost Function (MSE):** $J(w,b) = \frac{1}{2n} \sum_{i=1}^{n} (y^{(i)} - \hat{y}^{(i)})^2$
3.  **Gradient Calculation:** $\nabla J = \frac{1}{n} X^T (\hat{y} - y)$
4.  **Weight Update:** $w = w - \alpha \cdot \nabla J$

### Complexity & Notation Analysis

The key advantage of our approach is visible when analyzing the algorithmic complexity per step.

| Symbol | Definition | Context in this Project |
| :---: | :--- | :--- |
| **$N$** | Number of Samples | 5,000 (Cells) |
| **$P$** | Number of Features | 20,000 (Genes) |
| **$k$** | Number of Iterations | e.g., 1,000 steps |
| **$\alpha$** | Learning Rate | Step size (Hyperparameter) |

**Comparative Complexity:**
* **Analytical (SVD/OLS):** $O(N \cdot P^2)$ — *Quadratic/Cubic relative to Features.*
* **Gradient Descent:** $O(k \cdot N \cdot P)$ — *Linear relative to Features.*

By keeping the complexity linear with respect to $P$, Gradient Descent allows us to handle High-Dimensional Data ($P \gg N$) with significantly lower peak memory usage, as demonstrated in our benchmarks.

---

## Methodology & Experimental Design

To ensure a fair evaluation, I developed two testing environments using **Synthetic Data**. This allowed us to control the "Ground Truth" and measure how well the model recovers known biological or ecological signals.

### Scenario 1: Agro-Ecological Accuracy
* **Context:** Predicting crop yield based on variables like Rainfall, Soil pH, and Fertilizer type.
* **Challenge:** Handling categorical data and features with vastly different scales.
* **Validation:** Comparison of RMSE and weight recovery.

### Scenario 2: High-Dimensional Genomics (Stress Test)
* **Context:** Single-Cell RNA Sequencing (scRNA-seq) where expression is measured across thousands of genes.
* **Dataset:** $N=5,000$ cells and $P=20,000$ genes.
* **Challenge:** Peak memory management and training efficiency in an underdetermined system ($P \gg N$).

---

## Implementation Highlights

Beyond the core algorithm, this project tackles two critical challenges often overlooked in academic implementations but vital for production environments.

### Feature Scaling
Gradient Descent is highly sensitive to the scale of input features. If one variable ranges from 0-1 (e.g., Soil pH) and another from 0-1000 (e.g., Rainfall), the cost function contours become elongated ellipses. The optimizer "zigzags" across the valley, taking much longer to reach the global minimum. So, I implemented automatic **Z-Score Normalization** inside the `fit` method:

$$x' = \frac{x - \mu}{\sigma}$$
This transforms the error surface into a symmetric bowl (spherical contours), allowing the gradient to point directly towards the minimum for faster and stable convergence.


### Schema Consistency
A common failure mode in deployment occurs when the test set (or live data) lacks specific categorical levels present during training (e.g., the training set has "Fertilizer A, B, C", but a specific test batch only contains "A and B"). Instead of simple One-Hot Encoding which would result in mismatched matrix dimensions, I implemented a robust schema enforcement using `pandas.reindex` inside the `predict` method. This guarantees that the model always receives the exact feature structure it expects, filling missing dummy columns with zeros automatically to prevent dimensionality errors.

---

## Experimental Results & Benchmarking

### Scenario 1: Convergence and Precision
The model demonstrated a perfect convergence path. As shown in the dashboard, our custom model recovered the "Ground Truth" signal with high correlation, matching the precision of the analytical solution.

![Accuracy Dashboard](img/scenario1_accuracy_dashboard.png)
*Left: Cost function decreasing smoothly. Right: Predicted vs. Actual values showing high accuracy.*

### Scenario 2: Computational Efficiency
In the high-dimensional Genomics test, the iterative nature of Gradient Descent proved to be more resource-efficient.

![Performance Benchmark](img/scenario2_performance_benchmark.png)

* **Speed:** Even in pure Python, our vectorized implementation achieved a **~1.13x speedup** over Scikit-Learn in this specific high-dimensional layout ($P \gg N$).
* **Memory:** Our model achieved **~11.4% less peak RAM usage** by avoiding expensive matrix decompositions.

---

## Project Structure

```bash
LINEAR-REGRESSION-FROM-SCRATCH/
│
├── src/
│   ├── __init__.py
│   └── linear_regression_gd.py                      # Core Class
│
├── notebooks/
│   ├── agro_case_study.ipynb                      # Scenario 1: Accuracy Check
│   └── single_cell_genomics_stress_test.ipynb     # Scenario 2: Scalability
│
├── img/                                             # Benchmark Visualizations
├── requirements.txt                                 # Dependencies
└── README.md

```
---

## Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/pedro-mota-864084204/)
[![Email](https://img.shields.io/badge/Email-333333?style=for-the-badge&logo=gmail&logoColor=white)](mailto:pedroaamota@outlook.com)
