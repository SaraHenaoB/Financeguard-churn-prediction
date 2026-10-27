# FinanceGuardCustomer | Churn Prediction & Retention Analytics

## Project Context

FinanceGuard is a digital bank that has experienced exponential growth in recent years, reaching 10,000 active customers. However, management has detected a worrying increase in the customer churn rate, which currently stands at 20% annually, resulting in multi-million losses.
As a Data Scientist, the goal is to develop a comprehensive Machine Learning model to predict which customers are most likely to leave the bank. This model will enable the retention team to implement personalized strategies to retain at-risk customers, with the objective of reducing churn to 15%.

## Model Objective:
Decrease FinanceGuard's bank churn from 20% annually to 15% annually. A 5% reduction in churn.

#### _Concepts and Considerations_

##### *Bank Churn*

Bank churn refers to the customer attrition rate when customers stop using bank services or close their accounts within a given period. It is important for calculating the bank's loss of recurring revenue. It is calculated as:

                    Churn Rate = (lost customers in a period / total customers at the beginning of that period) X 100

Account closure can be voluntary, where the user decides to close their accounts, or involuntary, when the bank closes them due to non-payment. In this case, churn is considered to be any customer who left FinanceGuard and withdrew their balance money.

Customer Acquisition Cost:

            "According to global financial industry studies (such as those published by Bain & Company and Harvard Business Review), acquiring a new customer is between 5 and 25 times more expensive than retaining an existing one. Therefore, the development of this Machine Learning model has a strong financial justification, focusing on retention spending efficiency."[1]

According to the literature, it is relevant to determine if a customer is likely to close their accounts and be classified as churn to avoid significant financial losses. Therefore, an initial logistic regression model will be performed for this purpose, and other models will be built to see which one best fits this need.
In the data provided by FinanceGuard, an 80-20 class imbalance is observed, meaning that out of 10,000 records, 20% correspond to customers who left the bank. This affects the regression model since its accuracy will be biased and may give a false prediction by being trained with a larger amount of data from one class over the other; therefore, the evaluation of this model will focus on the RECALL metric.

Another reason to focus on this metric is that, for the business, a false negative is fatal compared to a false positive. This translates to the following:

For FinanceGuard, it is worse to predict that a customer will stay with the bank and then have them leave (False Negative for churn) without trying to retain them, than to classify a customer as potential churn and offer some financial benefit, only for them to remain with the bank anyway (False Positive).

In the first case, the model classifies the customer as non-churn (y=0), the customer leaves, and the bank loses liquidity as the customer withdraws their monetary balance. In the second case, the bank only absorbs the cost of the financial benefit offered to the customer who was expected to "leave" (who ultimately stays). Furthermore, this can generate customer satisfaction and secure their loyalty for longer.

Taking the data into account, the 20% customer loss represents a loss of € 185,588,094.63 in funding capacity and an average of € 91,108.54 for each churned customer. There is no information available on the interest rates FinanceGuard operates with, so losses are measured in capital loss and funding capacity.

## Executive Summary of the Project
Detailed metrics analysis, comparisons, and key conclusions of the models are documented in the official report:

👉 **[Read the Executive Summary](./docs/Executive%20Summary.md)**

## Repository Structure
The repository is organized as follows:

   ```
Financeguard-churn-prediction/
├── data/
│   ├── raw/
│   │   └── Churn_Modelling.csv
│   └── processed/
│       └── X_No escalado.csv
│       └── y.csv
│       └── X_test_final_escalado.csv
│       └── X_train_final_escalado.csv
│       └── y_test_final.csv
│       └── y_train_final.csv
└── docs/
│   └── Executive Summary.md                             # Executive summary and performance evaluation of algorithms
├── notebooks/
│   └── 1_EDA_RegresionLogistica.ipynb                   # EDA and baseline Logistic Regression model.
│   └── 2_GradientBoosting_Optimizacion.ipynb            # Models (Gradient Boosting, CatBoost) and hyperparameter optimization.
│   └── 3_AprendizajeNoSupervisado.ipynb                 # Analysis and segmentation using unsupervised algorithms.
└── powerbi/
│   └── Proyect FinanceGuard.pptx
└── README.pdf
   ```

## Technologies and Libraries Used
*   **Python 3.x**
*   **Jupyter Notebook**
*   **CatBoost / Scikit-Learn** (Predictive Modeling)
*   **Pandas / NumPy** (Data Manipulation)
*   **Matplotlib / Seaborn** (Visualization)

## How to Run the Project?

1. Clone this repository:
   ```bash
   git clone https://github.com/SaraHenaoB/Financeguard-churn-prediction.git
   cd Financeguard-churn-prediction
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks in chronological order (`1_EDA...`, `2_Gradient...`, `3_Apren...`) using Jupyter.

## Bibliography:
[1] Reichheld, F. F. (1990). "Zero Defections: Quality Comes to Services". Harvard Business Review (in collaboration with Bain & Company). 
Link: https://www.researchgate.net/publication/382441686_How_to_Multiply_Customers_if_You_Already_Have_Some_While_Lowering_Customer_Acquisition_Costs
