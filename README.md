# Logistic Regression — Binary Classification with Scikit-Learn

## 🧠 Problem Statement and Goal of Project

The objective of this notebook is to apply **Logistic Regression** for solving a binary classification task. The goal is to develop a clear understanding of the model workflow, implementation using scikit-learn, and model evaluation techniques.

This notebook is both a demonstration of skill and a learning artifact, showing not only the final results but also the process of building and validating a supervised learning model.

## 🔍 Solution Approach

* Load and prepare the dataset.
* Split data into training and testing sets.
* Fit a Logistic Regression model using `scikit-learn`.
* Predict on test data.
* Evaluate the model using accuracy score and comparison of predictions.

The emphasis is on understanding core ML concepts and evaluating model performance with clear code.

## 🛠 Technologies & Libraries Used

* Python (Jupyter Notebook)
* NumPy
* Pandas
* Matplotlib / Seaborn
* Scikit-learn (`LogisticRegression`, `train_test_split`, `metrics`)

## 📊 Dataset Description

The dataset includes multiple features and a binary target column named `Outcome`. Based on the code:

```python
X = data.drop(columns=['Outcome'])
y = data['Outcome']
```

The problem is framed as a **binary classification** task, where the model predicts one of two possible outcomes.

## ⚙️ Installation & Execution Guide

1. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. Launch Jupyter Notebook and open:

```bash
Logistic_Regression.ipynb
```

3. Run all cells to see training, prediction, and evaluation outputs.

## 📈 Key Results / Performance

* The model was trained on a split of the dataset using `train_test_split`.
* After prediction, the model achieved an accuracy of:

```python
The accuracy of Logistic Regression is: 0.9736842105263158
```

* ✅ This **97.4% accuracy** confirms that Logistic Regression is highly effective for this binary classification task when the dataset is clean and well-structured.

## 🖼️ Screenshots / Sample Outputs

Not included in this notebook, but sample outputs include:

* Accuracy score
* Predicted labels vs. actual
* Model evaluation via `sklearn.metrics`


* Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering. *

## 🧠 Additional Notes & Learning Outcomes

* Demonstrates knowledge of supervised learning pipelines.
* Shows familiarity with sklearn's `LogisticRegression` implementation.
* Reflects ability to execute, evaluate, and interpret classification models in practice.
* Some cells are exploratory or intentionally simple — aimed at reinforcing foundational concepts.

## 👤 Author

## mehran Asgari
## **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com).
## **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari).

---

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.