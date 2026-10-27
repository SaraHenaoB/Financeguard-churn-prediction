# Models Report – Proyect: FinanceGuard

## Executive Summary
FinanceGuard faces an annual churn rate of 20.37%. The objective is to be able to predict the customers most likely to leave and apply retention strategies that manage to reduce this Churn to 15%, that is, to at least be able to identify 5% of the customer candidates to churn. For this purpose, 3 progress updates were created where data cleaning and preparation (EDA) were developed, as well as the development of supervised models: Logistic Regression, Random Forest, XGBOOST, LightGBM, CatBoost, and Stacking (XGBOOST, LightGBM, CatBoost), and the development of unsupervised models KMeans and DBSCAN for customer segmentation and profile identification through dimensionality reduction techniques such as PCA (Principal Component Analysis) and t-SNE.

### Key Findings:
* **Financial Impact:**
    - Pérdida de capacidad de fondeo total anual: € 185,588,094.63
    - Pérdida de capacidad en promedio anual por cliente: € 91,108.54
* Optimization of the baseline Logistic Regression model through the Youden Index, increasing its Recall from 19% to 68% with a ranking capacity of 77% (ROC-AUC).
* **Best Supervised Model:** Stacking (XGBoost + LightGBM + CatBoost) with ROC-AUC 0.872 and Recall of 0.7838. Tied with LightGBM and only surpassed by a 1.48% difference in Recall. In financial terms, this difference represents the identification of € 2.7M per year.
* **Segmentation:** K-Means with K=3 identifies a high-risk cluster (Germany, 32.4% churn) compared to 16% in other segments. Identification of specific risk profiles with t-SNE that would contribute up to 1.48% in global churn reduction.
* **Key Variables:** Geography, Balance, NumOfProducts, IsActiveMember are the most important predictors in segmentation.



## Synthesis by Progress Update

### Progress Update 1 – Logistic Regression
Problem contextualization, initial exploration of the FinanceGuard dataset, estimation of losses due to churn, hypotheses of relationships between variables, preparation, encoding, and scaling of variables, training of Logistic Regression model, threshold optimization, and comparison.

#### Performance

<img width="975" height="731" alt="image" src="https://github.com/user-attachments/assets/ddb3c784-6e84-4423-a0c5-dfc5b272f4d3" />

* Dataset: 10,000 customers, 20.37% churn (80/20 imbalance).
* Variables: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary (with Geography and Gender encoding).

#### Coefficient Interpretability
* The coefficients with greater magnitude have a greater incidence on the probability of churn. In this case, the variables that increase the probability of churn the most are Age and being from Germany, while being an Active Member and Male gender retain the customer.
* The features with the least impact are Tenure and being from Spain.

#### Odds Ratio
- For each standard deviation increase in Age, the odds increase by 2.09, which is a 109.4% increase in the probability of leaving the bank.
- For each standard deviation increase in the number of people from Germany, the probability of leaving the bank increases 1.42 times, i.e., increases by 42.9%.
- For each standard deviation increase in Balance, the odds are multiplied by 1.17 times, i.e., churn probability increases by 17.42%.
- For each standard deviation increase in Active Members, the probability decreases by 40.3%.
- Given that Gender was encoded Female=0 (base) and Male=1, for each standard deviation increase in men using products, the probability of the customer leaving the bank decreases by 22.9%.
- **Intercept:** When all predictor variables are zero (which technically represents a customer with average age and balance, who is Female and lives in France), the actual probability of leaving the bank is barely 16.17%.

#### Strengths
- Easy to explain to the business (coefficients and odds ratios).
- Fast to train and deploy.
- Provides calibrated probabilities to prioritize actions.
- Serves as a benchmark for more complex models.

#### Limitations
- Linear relationships in log-odds space; may underestimate interactions if threshold is not adjusted.
- Moderate Recall with 0.5 threshold: a portion of churners is missed.
- Does not capture non-linear patterns well, which trees and boosting do leverage.

---

### Progress Update 2 - Gradient Boosting and Optimization
All models took class imbalance into account and set the parameter to 'balanced' or class ratio (3.91) to penalize errors in the churn class.

#### Best Boosting Model Identified

<img width="975" height="185" alt="image" src="https://github.com/user-attachments/assets/631545f6-2dde-415e-b42d-f206f2fd15a2" />

After testing and fine-tuning all advanced models, and having such similar ROC-AUC values, we decided to choose based on our metric of interest, which is Recall.
Based on this, the Stacking Ensemble is the winner and the model the bank should use.

Although in the theoretical capacity to rank customers by risk (ROC-AUC) it was practically tied with LightGBM (both with 0.872), Stacking slightly outperformed it by 1.48% in the Recall metric. For a decision threshold of 0.5, it achieved the highest Recall in the entire project (78.38%). This means it catches 78 out of every 100 customers who will actually leave the bank, stopping the multi-million loss identified in Progress Update 1 (€ 91,108.54 on average per customer). Furthermore, a positive aspect is that with a Precision of 49.92%, half of the alerts for potential churners will be real, which the bank can manage.

The difference between using the first or second model for the business would be in that 1.48% difference in Recall, which equals 30 customers (total churners: 2,037). In financial terms, this means identifying € 2,746,703.82 per year.

#### Optimal Parameters (LightGBM)
- `n_estimators=300`
- `num_leaves=5`
- `max_depth=5`
- `learning_rate=0.05`
- `scale_pos_weight=3.91`
- `random_state=42`
- `verbose=-1`

#### Feature Importance of the Best Model (LightGBM / Stacking)
Importance is assigned according to the frequency it was used to create splits; higher usage means higher importance.
1. **Age** – Age as primary risk predictor.
2. **Balance** – Account balance.
3. **NumOfProducts** – Number of products (1 or 4 are riskier).
4. **EstimatedSalary**
5. **CreditScore**
6. **Geography** (Germany/Spain) – Market.
7. **IsActiveMember** – Customer activity.
8. **Tenure, Gender** – Minor but non-zero contribution.
9. **HasCrCard** had zero contribution.

#### Performance Gain vs. Baseline Model

| Metric | Logistic Regression (Progress Update 1) | LightGBM / Stacking (Progress Update 2) | % Improvement |
| :--- | :--- | :--- | :--- |
| ROC-AUC | 0.7748 | 0.8728 | 12.65% |
| Recall | 0.19 | 0.7838 | 312.53% |
| Precision | 0.59 | 0.4992 | -15.39% |
| F1-Score | 0.28 | 0.6099 | 117.82% |
| Accuracy | 0.81 | 0.796 | -1.73% |
| Threshold | 0.5 | 0.5 | 0% |

Stratified Cross-Validation (5-fold) with ROC-AUC metric ensures stable evaluation on imbalanced data.

---

### Progress Update 3 - Unsupervised Learning

#### Customer Segments Identified
- **Cluster 0:** Germany, balance 119,730.12, churn rate 32.4%
- **Cluster 1:** France, balance 62,092.64, churn rate 16.2%
- **Cluster 2:** Spain, balance 61,818.15, churn rate 16.7%

<img width="975" height="415" alt="image" src="https://github.com/user-attachments/assets/6ffadd75-8d0e-47aa-aff9-64018efc2086" />

<img width="664" height="438" alt="image" src="https://github.com/user-attachments/assets/e7a06314-4b1e-4dfe-b236-06ec517f0bd5" />

**PCA Visualization Analysis:**
- **Purple Group (Germany):** Completely separates to the right due to high account balances.
- **Green Group (France) & Yellow Group (Spain):** Appear mixed in the center because they share identical age, salary, and product characteristics. The graph visually confirms the low silhouette score. Data overlaps continuously, except for the Germany group which segments clearly.

#### t-SNE

<img width="738" height="490" alt="image" src="https://github.com/user-attachments/assets/fe75fd96-a866-4a7e-beaf-0f378580a5e3" />

Dimensionality reduction shows that the 11 variables have a non-linear relationship that could not be captured with PCA. Small islands of local clustering are observed for each country. This also supports the results obtained with DBSCAN when increasing the epsilon search radius. For a search radius of half a standard deviation, it found 14 groups, but increasing the radius produced 48 groups.

<img width="732" height="490" alt="image" src="https://github.com/user-attachments/assets/63b700fb-5d06-4973-8cdd-9b7ef174b9d7" />

Local groups marking high churn risk trends can be identified and targeted specifically, complementing the cluster segmentation strategy. For example:

<img width="544" height="656" alt="image" src="https://github.com/user-attachments/assets/97363c03-ac9a-4abb-a5ed-f9b1f1a77b4a" />

#### Business Insights by Cluster
- **Cluster 0:** Focus priority for retention strategies, review experience in Germany and possible product/pricing adjustments. Total customers: 25.09% (2,509) with 32.4% churn, contributing 8.13% of the overall 20% churn.
- **Clusters 1 and 2:** Being below 20% Churn, they are stable, useful for growth strategies, benchmarking, and cross-selling without neglecting basic retention.
- **DBSCAN** on this dataset generated many outliers (homogeneous data); K-Means proved more useful for actionable segmentation.
- **t-SNE** can complement K-Means segmentation. Acting on specific groups, for example: selecting one of the islands visually showing a higher churn concentration located between t-SNE1 [25-50] and t-SNE2 [50-75] coordinates, which has a local churn rate of 53% for an island of 277 customers. This would end up contributing 1.47% toward the overall 5% churn reduction goal.

#### Features Derived from Clustering
- `'cluster_kmeans'`: Cluster label (0, 1, 2) for use in supervised models or business rules.
- `'distancia_centroide'`: Euclidean distance from each customer to their assigned centroid.
- `'nivel_riesgo_cluster'`: Customer risk level label: 'Alto Riesgo', 'Medio Riesgo', 'Bajo Riesgo'.

---

## Lessons Learned

### When to Use Supervised vs. Unsupervised Models

| Criteria | Supervised Models (Prediction) | Unsupervised Models (Segmentation) |
| :--- | :--- | :--- |
| **Main Objective** | Predict the probability of churn for each customer. | Discover hidden behavioral patterns and groups. |
| **Data Requirement** | Historical data with clear labels (0 = Active, 1 = Churn). | Behavioral data without the need for labels. |
| **Common Algorithms** | XGBoost, Random Forest, Logistic Regression. | K-Means, DBSCAN. |
| **Success Metrics** | F1-Score, AUC-ROC, Recall (Sensitivity). | Silhouette Score, Inertia (Elbow method). |
| **Business Action** | Direct retention campaigns to at-risk customers. | Design differentiated strategies by customer type. |