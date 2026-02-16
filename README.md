# 📉 Linear Regression from Scratch using Gradient Descent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **A vectorized, modular implementation of Linear Regression using Gradient Descent, benchmarked against Scikit-Learn in high-dimensional biological scenarios.**

---

## Project Overview

While `sklearn.linear_model.LinearRegression` is the industry standard, its reliance on the Ordinary Least Squares (OLS) closed-form solution ($O(P^3)$) creates significant memory bottlenecks when handling wide datasets. 

To address this, this project explores the use of **Gradient Descent** as an iterative alternative. Instead of attempting to solve the entire system at once through costly matrix inversions, Gradient Descent approaches the optimal solution in steps. This study aims to demonstrate how this iterative process can be more resource-efficient in high-dimensional contexts, providing a practical way to manage time and memory usage without sacrificing the model's ability to learn from the data.

### Key Features
* **Vectorized Implementation:** Fully optimized matrix operations using NumPy (no slow `for` loops).
* **Automatic Feature Engineering:** Built-in Z-Score Normalization and One-Hot Encoding handling.
* **Dual-Scenario Validation:** Benchmarked using generated **Synthetic Data** across two distinct domains:
    * **Agro-Ecological Case:** Verified convergence and prediction accuracy against Scikit-Learn.
    * **Single-Cell Genomics Case:** Stress-tested memory efficiency on high-dimensional data ($N=5,000, P=20,000$).
* **Scikit-Learn Compatible:** Follows the standard `.fit()` and `.predict()` API design.

---

## Theoretical Foundation & Algorithm

### Gradient Descent vs. Analytical Solution
The standard OLS approach solves the regression problem by finding the point where the derivative of the cost function is zero using the **Normal Equation**:
$$\theta = (X^T X)^{-1} X^T y$$

However, inverting the matrix $(X^T X)$ has a computational complexity of approximately **$O(P^3)$**. In our Genomics scenario ($P=20,000$), this becomes a massive bottleneck. 



Our implementation uses **Batch Gradient Descent**, which optimizes the weights iteratively:
1.  **Prediction:** $\hat{y} = Xw + b$
2.  **Cost Function (MSE):** $J(w,b) = \frac{1}{n} \sum (y - \hat{y})^2$
3.  **Gradient Update:** $w = w - \alpha \frac{\partial J}{\partial w}$

By using an iterative approach, we reduce the complexity per iteration to **$O(k \cdot n \cdot p)$**, making it significantly more memory-efficient for wide datasets.

---

## Methodology & Experimental Design

To ensure a fair evaluation, we developed two testing environments using **Synthetic Data**. This allowed us to control the "Ground Truth" and measure how well the model recovers known biological or ecological signals.

### Scenario 1: Agro-Ecological Accuracy
* **Context:** Predicting crop yield based on variables like Rainfall, Soil pH, and Fertilizer type.
* **Challenge:** Handling categorical data and features with vastly different scales.
* **Validation:** Comparison of RMSE and weight recovery.

### Scenario 2: High-Dimensional Genomics (Stress Test)
* **Context:** Single-Cell RNA Sequencing (scRNA-seq) where expression is measured across thousands of genes.
* **Dataset:** $N=5,000$ cells and $P=20,000$ genes.
* **Challenge:** Peak memory management and training efficiency in an underdetermined system ($P \gg N$).

---

## Experimental Results & Benchmarking

### Scenario 1: Convergence and Precision
The model demonstrated a perfect convergence path. As shown in the dashboard, our custom model recovered the "Ground Truth" signal with high correlation, matching the precision of the analytical solution.

![Accuracy Dashboard](img/scenario1_accuracy_dashboard.png)
*Left: Cost function decreasing smoothly. Right: Predicted vs. Actual values showing high accuracy.*

### Scenario 2: Computational Efficiency (The "Win")
In the high-dimensional Genomics test, the iterative nature of Gradient Descent proved to be more resource-efficient.

![Performance Benchmark](img/scenario2_performance_benchmark.png)

* **Memory Win:** Our model achieved **~10% less peak RAM usage** by avoiding expensive matrix decompositions.
* **Speed:** Even in pure Python, our vectorized implementation achieved a **1.44x speedup** over Scikit-Learn in this specific high-dimensional layout ($P \gg N$).

---

## Project Structure

```bash
LINEAR-REGRESSION-FROM-SCRATCH/
│
├── src/
│   ├── __init__.py
│   └── linear_regression_gd.py   # Core Class
│
├── notebooks/
│   ├── 1_agro_case_study.ipynb            # Scenario 1: Accuracy Check
│   └── 2_single_cell_genomics_stress_test.ipynb # Scenario 2: Scalability
│
├── img/                          # Benchmark Visualizations
├── requirements.txt              # Dependencies
└── README.md

```
---

## Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/pedro-mota-864084204/)
[![Email](https://img.shields.io/badge/Email-333333?style=for-the-badge&logo=gmail&logoColor=white)](mailto:pedroaamota@outlook.com)
