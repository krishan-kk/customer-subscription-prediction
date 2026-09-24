# Customer Subscription Renewal Prediction

End-to-end machine learning pipeline that predicts whether a customer will make another purchase on a subscription-based audiobook platform, using a neural network trained on customer purchase and engagement data.

## Business Problem

Subscription businesses need to know, in advance, which customers are likely to renew or purchase again so they can focus marketing spend and retention efforts on the right customers instead of contacting everyone equally. This project builds a binary classification model that predicts customer renewal from historical purchase and engagement behavior, supporting **targeted marketing** and **customer retention** decisions.

## Dataset

- **Source:** Simulated data representing 2 years of customer engagement on a fictional audiobook app. Each row represents one customer.
- **Size:** 14,084 customers, 10 input features
- **Features:** overall and average audiobook length purchased, overall and average price paid, whether the customer left a review, average review score, total minutes listened, completion rate, number of support requests, and days between last visit and purchase date
- **Target:** binary whether the customer made another purchase (~15.9% positive class, so the raw data is imbalanced)

_Note: this is simulated data, not real customer records from an actual company._

## Data Preprocessing (`Booksubscription_data_process.ipynb`)

- Loaded the raw CSV with NumPy
- Balanced the dataset by undersampling the majority class to match the minority (renewed) class, addressing the ~16% class imbalance
- Standardized all input features with `sklearn.preprocessing.scale`
- Shuffled the balanced data and split it into training, validation, and test sets (80/10/10), saved separately as `subscriptions_data_train.npz`, `subscriptions_data_validation.npz`, and `subscriptions_data_test.npz`

No exploratory data analysis or feature engineering was performed beyond scaling — all 10 original features are used as-is.

## Model (`Subscription_model.ipynb`)

- Feedforward neural network built with TensorFlow/Keras, architecture chosen manually:
  - 2 hidden layers, 50 units each, ReLU activation
  - Softmax output layer for binary classification
- Compiled with the Adam optimizer and sparse categorical cross-entropy loss
- Trained with early stopping (patience = 5, restoring best weights) to prevent overfitting, monitored on the validation set
- Evaluated once on the held-out test set

## Results

- **Test accuracy: 83.26%**
- In plain terms: the model correctly predicts whether a customer will re-purchase in roughly 5 out of 6 cases in this test set.

_Only accuracy is currently reported precision, recall, and a confusion matrix would give a fuller picture, especially since the original (pre-balancing) data is imbalanced. This is listed as a planned improvement below._

## Future Improvements

- Test on a larger and/or real-world dataset
- Add a reusable scoring module that takes new/external customer data as input and outputs renewal predictions, rather than only working within the notebook

## Tech Stack

Python, NumPy, Scikit-learn, TensorFlow/Keras

## Project Structure

```
├── raw_customer_data.csv
├── subscriptions_data_train.npz
├── subscriptions_data_validation.npz
├── subscriptions_data_test.npz
├── Booksubscription_data_process.ipynb
├── Subscription_model.ipynb
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run `Booksubscription_data_process.ipynb` to preprocess the raw data and generate the `.npz` files.
3. Run `Subscription_model.ipynb` to train and evaluate the model.
