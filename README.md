# CreditCardFraudDetection
This project demonstrates a credit card fraud detection system using a Decision Tree model. The dataset is balanced using SMOTE, and the model is evaluated using cross-validation with a Random Forest model and ROC curves.

## Project Description

Credit card fraud detection is a significant issue for financial institutions. This project aims to build a machine learning model to detect fraudulent transactions. We use a dataset from Kaggle, preprocess the data, balance it using SMOTE, and train a Decision Tree model. The model's performance is evaluated using cross-validation with a Random Forest and ROC curves.

## Dataset

The dataset used for this project is available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains transactions made by credit cards in September 2013 by European cardholders. The dataset presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions.

## Installation

To install the necessary packages, run:

```sh
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

    ```sh
    git clone https://github.com/your-username/CreditCardFraudDetection.git
    ```

2. Navigate to the project directory:

    ```sh
    cd CreditCardFraudDetection
    ```

3. Run the Jupyter notebook or Python scripts to train the model and make predictions:

    - Open the Jupyter notebook:

      ```sh
      jupyter notebook notebooks/credit_card_fraud_detection.ipynb
      ```
## Project Workflow

### Step 1: Data Loading and Preprocessing

- **Load the dataset**: The dataset is loaded into a Pandas DataFrame.
- **Data scaling**: The 'Amount' column is scaled using `StandardScaler`.

![Data Loading and Preprocessing](path/to/screenshot1.png)
