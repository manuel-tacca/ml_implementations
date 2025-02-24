# Machine Learning Models

This project implements basic machine learning models from scratch, including **Logistic Regression** and **Linear Regression**, using **Gradient Descent**.

## Project Structure
```
machine_learning/
│── logistic_regression/
│   ├── src/
│   │   ├── logistic_regression.py  # Logistic Regression implementation
│   ├── test/
│   │   ├── test_logistic_regression.py  # Unit tests for logistic regression
│
│── linear_regression/
│   ├── src/
│   │   ├── linear_regression.py  # Linear Regression implementation
│   ├── test/
│   │   ├── test_linear_regression.py  # Unit tests for linear regression
│
│── data/
│   ├── get_data.py  # Functions to generate synthetic datasets
│
│── main.py  # Runs all tests
│── README.md  # Documentation
│── requirements.txt  # Dependencies
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ml_implementations.git
   cd ml_implementations
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Running Logistic Regression
```python
from logistic_regression.src.logistic_regression import LogisticRegressionGD
from data.get_data import get_classification_data

X, y = get_classification_data()
model = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)
print(predictions)
```

### 2. Running Linear Regression
```python
from linear_regression.src.linear_regression import LinearRegressionGD
from data.get_data import get_regression_data

X, y = get_regression_data()
model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)
print(predictions)
```

## Running Tests
To run all unit tests:
```bash
python main.py
```

Or manually run:
```bash
python -m unittest discover -s .
```

## Dependencies
This project requires:
- Python 3.x
- NumPy
- Unittest (built-in)

Install dependencies with:
```bash
pip install -r requirements.txt
```
