# Statistical Analysis & Predictive Modeling on Titanic Survival

[![R](https://img.shields.io/badge/Language-R%20%3E%3D%204.0-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-titanic__train-orange.svg)](https://cran.r-project.org/package=titanic)

An end-to-end inferential statistics and predictive modeling pipeline built in R using the RMS Titanic dataset (`titanic::titanic_train`). This study evaluates non-parametric distributional assumptions, performs bivariate association tests, fits a binary logistic regression model using $k$-fold cross-validation, and provides clinical model diagnostic evaluations.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dataset Summary](#-dataset-summary)
- [Data Preprocessing & Imputation](#-data-preprocessing--imputation)
- [Exploratory Statistical Testing](#-exploratory-statistical-testing)
- [Predictive Modeling Framework](#-predictive-modeling-framework)
- [Diagnostics & Performance Metrics](#-diagnostics--performance-metrics)
- [Visualizations](#-visualizations)
- [Installation & Reproducibility](#-installation--reproducibility)
- [Key Insights & Recommendations](#-key-insights--recommendations)

---

## 📌 Project Overview

The objective of this project is twofold:
1. **Inferential Hypothesis Testing:** Statistically test whether demographic profiles (e.g., sex, passenger class, ticket fare) were significant discriminators of survival likelihood during the maritime disaster.
2. **Supervised Classification:** Train and diagnose a regularized binary Generalized Linear Model (Logistic Regression) to accurately predict passenger survival on out-of-sample data, validating with cross-validation and Area Under the ROC Curve (AUC).

---

## 📂 Repository Structure

```text
titanic-statistical-analysis-r/
├── analysis.R                  # End-to-end reproducible R script
├── plots/                      # Generated high-resolution diagnostic graphics
│   ├── diagnostic_residuals.png
│   ├── roc_curve.png
│   └── variable_importance.png
├── .gitignore                  # R project build and temporary files ignore list
└── README.md                   # Repository documentation and report
