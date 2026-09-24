# Customer Subscription Renewal Prediction

End-to-end machine learning pipeline that predicts whether an audiobook-service customer will make another purchase, using neural network classification on customer purchase and engagement data.

## Problem

The business collects data on customer purchases and engagement with an audiobook platform but has no way to identify, in advance, which customers are likely to return. This project builds a supervised classification model to predict customer renewal, supporting **targeted marketing** and **customer retention** efforts.

## Dataset

- **Source:** Raw customer data (`raw_customer_data.csv`), 14,084 customers
- **Features (10):** overall and average book length purchased, overall and average price paid, whether the customer left a review, average review score, total minutes listened, completion rate, number of support requests, and days between last visit and purchase date
- **Target:** binary — whether the customer made another purchase (~15.9% positive class, heavily imbalanced)

## Approach

**1. Data Preprocessing** (`Booksubscription_data_process.ipynb`)
- Loaded raw CSV data with NumPy
- Balanced the dataset by undersampling the majority class to match the minority class count, addressing the ~16% positive class imbalance
- Standardized input features using `sklearn.preprocessing.scale`
- Shuffled and split the data into training, validation, and test sets (80/10/10)
- Saved the processed splits as `.npz` files for reuse

**2. Model Training** (`Subscription_model.ipynb`)
- Built a feedforward neural network with TensorFlow/Keras: two hidden layers (50 units each, ReLU activation) and a softmax output layer for binary classification
- Compiled with the Adam optimizer and sparse categorical cross-entropy loss
- Trained with early stopping (patience = 5) on the validation set to prevent overfitting
- Evaluated on a held-out test set

## Results

- **Test accuracy: 83.26%**

## Tech Stack

- Python, NumPy, Scikit-learn, TensorFlow/Keras

## Project Structure

```
├── data/
│   ├── raw_customer_data.csv
│   ├── subscriptions_data_train.npz
│   ├── subscriptions_data_validation.npz
│   └── subscriptions_data_test.npz
├── notebooks/
│   ├── Booksubscription_data_process.ipynb
│   └── Subscription_model.ipynb
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run `notebooks/Booksubscription_data_process.ipynb` to preprocess the raw data and generate the `.npz` files.
3. Run `notebooks/Subscription_model.ipynb` to train and evaluate the model.
