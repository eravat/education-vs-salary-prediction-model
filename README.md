# Education vs. Salary Prediction — Linear Regression from Scratch

A univariate linear regression model, implemented from scratch using gradient descent (no scikit-learn), predicting yearly salary based on years of education.

## Overview

This project was built while working through Andrew Ng's Machine Learning Specialization, as a way to implement gradient descent hands-on rather than relying on a library's built-in `.fit()` method. The goal was to understand — and be able to explain — exactly what's happening under the hood: the cost function, the partial derivatives, and how the weight (`w`) and bias (`b`) update on each iteration.

## Dataset

- **Source:** U.S. Census (CPS) earnings and education data, adapted from Stock & Watson's *Introduction to Econometrics* teaching dataset.
- **Features used:** `education` (years of schooling)
- **Target:** `yearly_salary` (converted from hourly earnings, assuming a 40-hour week / 50-week year)
- **Note:** the dataset is restricted to workers aged 29–30, so the model reflects that specific age band rather than the general population.

## Method

The model fits a straight line of the form:

```
f(x) = w * x + b
```

where `x` is years of education and `f(x)` is predicted yearly salary. `w` and `b` are learned via **batch gradient descent**:

1. Compute the cost function `J(w, b)` — the mean squared error between predictions and actual salaries.
2. Compute the partial derivatives `∂J/∂w` and `∂J/∂b`.
3. Simultaneously update `w` and `b` using the learning rate `α`.
4. Repeat until convergence.

No `scikit-learn` or other ML library is used for the model itself — only NumPy for array/vector operations.

## Files

- `gradient_descent.py` — main script: loads data, runs gradient descent, prints/plots results
- `[your CSV filename]` — the dataset (education, yearly_salary columns)
- `requirements.txt` — Python dependencies

## How to run

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
python gradient_descent.py
```

## What I learned / limitations

- It takes much longer than I anticipated to minimise the cost function due to the large volume of data and high amounts of iterations required, to the point where I decided not to spend time on it any longer, mainly because it's unlikely to teach me anything new. I also learnt that a decrease in the learning rate can lead to a substantially more accurate result, and can prevent divergence, but this requires a large number of iterations which takes a much longer time.

## Next steps

- Extend to multiple linear regression (education + age + experience) after Week 2 of the ML Specialization
- Compare results against scikit-learn's `LinearRegression()` to sanity-check the from-scratch implementation
