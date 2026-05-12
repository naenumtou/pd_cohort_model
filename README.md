# IFRS 9 PD Modeling Framework via Cohort Approach

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-purple?logo=pandas&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-green?logo=numpy&style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-Scientific%20Computing-blue?logo=scipy&style=for-the-badge)
![statsmodels](https://img.shields.io/badge/statsmodels-Statistical%20Modeling-red?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal?style=for-the-badge)
![MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

This repository implements a **cohort-based Probability of Default (PD)** model aligned with IFRS 9 Expected Credit Loss (ECL) Requirements. The model estimates **lifetime marginal PD(s)** by tracking default behavior across origination cohorts and time since origination. It is designed for use in Stage 1, Stage 2, and Stage 3 (Impaired = PD 100%) calculations and supports transparent, auditable credit risk modeling suitable for regulatory and financial reporting.

<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/29ad6c83-2be6-4b08-b00f-4f8eff2a14f9" />
</p>

## Overview
This project implements a **Cohort PD model** designed to support IFRS 9 Expected Credit Loss (ECL) Calculation. The model estimates cumulative and marginal for the lifetime PD by tracking default behavior over time for exposures grouped by origination cohort. By observing how default rates evolve with time since origination (observed), the approach provides a transparent and interpretable framework that aligns well with IFRS 9 requirements for forward-looking credit risk estimation.


The implementation emphasizes:
- Cohort-level transparency for auditability and model governance
- Vectorized numerical computation for efficiency and scalability
- Flexible aggregation across segments using exposure-based or observation-based weights


The resulting PD term structures can be directly used in Stage 1 and Stage 2 ECL Calculation. The project is intended to serve as a practical reference implementation for credit risk practitioners, model developers, and validators, rather than a black-box model. All calculations are made explicit, facilitating validation, backtesting, and model explainability.

## Project Structure
```
pd_cohort_model/
├── models/          #Trainned model and parameters (pkl.)
│   ├── actual_cumulative_odr.pkl
│   ├── chain_cumulative_odr.pkl
│   ├── w_avg_chain_cumulative_odr.pkl
│   ├── w_avg_gamma_cumulative_odr.pkl
│   ├── w_avg_gamma_parameters.pkl
│   ├── unbias_cumulative_odr.pkl
│   ├── fwl_model.pkl
│   └── pit_cumulative_lifetime_pd.pkl  
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_chaid_segmentation.ipynb
│   ├── 03_base_cohort.ipynb
│   ├── 04_fwl_model.ipynb
│   └── 05_lifetime_calibration.ipynb
├── src/
│   ├── data_prep.py
│   ├── segment_support.py
│   ├── base_builder.py
│   ├── regression_model.py
│   ├── curve_calibration.py
│   ├── stats_testing.py
│   └── plot_function.py
├── data/          
│   ├── processed/
|   |   ├── train_data.parquet          #Not tracked by git
|   |   ├── cohort_count.parquet
|   |   ├── unbias_odr.parquet
|   |   ├── monthly_odr.parquet
|   |   └── mev_transformed.parquet
│   └── raw/
|   |   ├── usedcar_transaction_score.parquet          #Not tracked by git
|   └── └── mev_data.csv
├── requirements.txt
└── README.md
```

## Project Details
### 0. Model Segmentation
> Note: The model segementation is only illustrative proposed. The full annlysis will be performed in another repository.

<div align="center">
 
| MOB | DPD | B-Score | Segment |
|:---:|:---:|:---:|:---:|
| <=6 | 0 |  | segment_0 |
| <=6 | 1-30 |  | segment_1 |
| <=6 | 31-60 |  | segment_2 |
| <=6 | 60-90 |  | segment_3 |
| >6 |  | B1 | segment_4 |
| >6 |  | B2 | segment_5 |
| >6 |  | B3 | segment_6 |
| >6 |  | B4 | segment_7 |
| >6 |  | B5 | segment_8 |
| >6 |  | B6 | segment_9 |
| >6 |  | B7 | segment_10 |
| >6 |  | B8 | segment_11 |

</div>

### 1. Unbias Model
<p align="center">
<img width="1691" height="930" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/6631967e-8cf0-4308-95d6-fce603b22a7b" />
</p>

#### 1.1 Cohort Building
**Cohort Based Default Measurement:** Exposures are segmented into homogeneous cohorts defined at observation point (e.g., vintage, product type, risk band). For each cohort, default events are tracked over time since observation point to construct marginal and cumulative default "triangles" forming the empirical basis for lifetime PD estimation under an IFRS 9 consistent default definition.
<p align="center">
 <img width="1632" height="759" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/7bde8a5c-04f7-4b2f-9c2b-9eb3a5b9e734" />
</p>

#### 1.2 Chain-Ladder
**Chain‑Ladder Pattern Projection:** Observed cohort default triangles are now run-off triangles. Given this fact that the more recent period, the less lifetime windows for tracking default. To fill the triangles, using chain‑ladder (development factor) methodology to extrapolate incomplete default histories. This produces projected ultimate cumulative lifetime PD for each cohort, enabling consistent estimation even for recently originated (immature) portfolios.
<p align="center">
<img width="1632" height="759" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/34bc2ebd-7790-4c73-9625-de97f9e98426" />
</p>

#### 1.3 Gamma Fitting
> Note: The Gamma distribution can be replaced by other statistic disctributions such as Weibull distribution. Or even mathematical formula e.g., Nelson Siegel but it needs to transform into correct basis. In this repository, the Gamma distribution is leveraged.

**Parametric Model via Gamma Distribution:** A Gamma distribution is fitted to the projected (extended) cumulative PD term structure for each cohort to remove sampling noise, enforce monotonicity, and obtain a smooth, stable PD curve. In the step, the segments from cohort built might be groupped as a **pool** level in case those segments are unable to build a stable curve by its own.

```
[Gamma distribution parameters]
Pool 0: Segment - ('segment_0',)
    Alpha: 1.7143     Beta: 1.0866     Constant: 0.0902
Pool 1: Segment - ('segment_1', 'segment_2', 'segment_3')
    Alpha: 0.8335     Beta: 1.2917     Constant: 0.3896
Pool 2: Segment - ('segment_4',)
    Alpha: 2.9373     Beta: 0.7116     Constant: 0.0129
Pool 3: Segment - ('segment_5',)
    Alpha: 3.1115     Beta: 0.6368     Constant: 0.0120
Pool 4: Segment - ('segment_6',)
    Alpha: 3.2286     Beta: 0.5826     Constant: 0.0220
Pool 5: Segment - ('segment_7', 'segment_8')
    Alpha: 2.4024     Beta: 0.6291     Constant: 0.0717
Pool 6: Segment - ('segment_9',)
    Alpha: 1.2083     Beta: 0.9439     Constant: 0.2240
Pool 7: Segment - ('segment_10', 'segment_11')
    Alpha: 0.6687     Beta: 1.1028     Constant: 0.4238
```

<p align="center">
<img width="1990" height="789" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/a15caf28-afe1-4d51-b946-c9a865c426d6" />
</p>

The Kolmogorov–Smirnov (K‑S) test is used to assess how well a dataset fits a specified theoretical distribution. During model development, two separate K‑S tests were performed to evaluate the fit of the Gamma function to the lifetime PD experience using PD Pool.

```
[KS Test]
Pool 0: Segment - ('segment_0',)
n: 7 KS-Stat: 0.1429 D-Critical: 0.483 Result: Pass
Pool 1: Segment - ('segment_1', 'segment_2', 'segment_3')
n: 6 KS-Stat: 0.1667 D-Critical: 0.519 Result: Pass
Pool 2: Segment - ('segment_4',)
n: 6 KS-Stat: 0.1667 D-Critical: 0.519 Result: Pass
Pool 3: Segment - ('segment_5',)
n: 6 KS-Stat: 0.1667 D-Critical: 0.519 Result: Pass
Pool 4: Segment - ('segment_6',)
n: 6 KS-Stat: 0.1667 D-Critical: 0.519 Result: Pass
Pool 5: Segment - ('segment_7', 'segment_8')
n: 6 KS-Stat: 0.1667 D-Critical: 0.519 Result: Pass
Pool 6: Segment - ('segment_9',)
n: 6 KS-Stat: 0.1667 D-Critical: 0.519 Result: Pass
Pool 7: Segment - ('segment_10', 'segment_11')
n: 6 KS-Stat: 0.1667 D-Critical: 0.519 Result: Pass
```

#### 1.4 Unbias Calibration
**Unbias Calibration:** The smoothed (Gamma) PD(s) are calibrated to align with long-run (TTC) Observed Default Rate (ODR). This is to ensure the key **Unbias** concept of IFRS 9 that no structural optimism or conservatism in the PD Estimated. The calibration is based on the concept that ratio of odds ratio for month m or year y and 12 months or 1-year will remain the same shape for segmentation level and the lifetime pool level. The equation below is for unbias calibration of odds function:

$$
Unbias\ lifetime\ ODR = 
\frac{
\text{ODR}_{\text{Unbias}}
\cdot
\frac{\text{ODR}_{TTC}}{\text{ODR}_{Target}}
}{
\text{ODR}_{\text{Unbias}}
\cdot
\frac{\text{ODR}_{TTC}}{\text{ODR}_{Target}}
+
\left(1-\text{ODR}_{\text{Unbias}}\right)
\cdot
\frac{1-\text{ODR}_{TTC}}{1-\text{ODR}_{Target}}
}
$$

where;

- ${\text{ODR}}_{\text{Unbias}}$ is a 12-months ODR from each segmentation;
- ${\text{ODR}}_{\text{Target}}$ is a ODR at month of 12 when monthly level or at year of 1 when yearly level for a corresponding to the lifetime pool;
- ${\text{ODR}}_{\text{TTC}}$ are TTC PD(s) in month m or year y for a corresponding to the lifetime pool

<p align="center">
<img width="1990" height="789" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/f4031e4f-d61e-46cc-aca3-dacb9bcdd45d" />
</p>

### 2. Forward-looking Model
<p align="center">
<img width="1690" height="931" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/fe580c69-d4d4-4728-99f2-ee073f5d786f" />
</p>

#### 2.1 Observed Default Rates (ODR)
12-months Observed Default Rates (ODR) are employed in a linear regression framework to quantify and analyze the relationship between ODR and the macroeconomic variables. The use of a 12‑month observation window helps smooth short‑term volatility and captures underlying credit risk dynamics, thereby providing a more stable and representative measure of default behavior for assessing macroeconomic sensitivity.

The historical ODR(s) are transformed using a logit function. The logit function converts continuous variables bounded between 0 and 1 into an unbounded (infinite) scale. This transformation is commonly applied to default rates to expand the range of the dependent variable, thereby enhancing its sensitivity and responsiveness in linear regression modeling. As a result, linear regression is preferred over logistic regression, as it allows for a broader set of established statistical tests to assess the model’s technical robustness and overall goodness of fit.
 
<p align="center">
<img width="989" height="593" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/40a4b639-9fa2-456b-8454-dda17f9340fb" />
</p>

#### 2.2 Macroeconomics Variables Transformation
A set of 24 macroeconomic variables (MEV) is used in the forward-looking model.The expected intuitive direction of their correlation with default rates for the portfolio. The intuitive direction reflects the anticipated relationship between changes in macroeconomic conditions and changes in default rates. For instance, an increase in the unemployment rate is expected to lead to higher default rates, implying a positive correlation.

<div align="center">
 
| No. | Macroeconomics variables | MEV | Sign with default | Reasons | Data type |
|:---:|---|:---:|:---:|---|:---:|
| 1 | Gross Domestic Product | GDP | Negative | The gross domestic product (GDP) is one of the primary indicators used to gauge the health of a country's economy. Therefore, an increase in GDP is expected to decrease in default rate. | Flow |
| 2 | Foreign Direct Investment  | FDI | Negative | A high Net FDI indicates good economic condition in Thailand as open economies with good growth prospects attract large amounts of FDI. Hence, net FDI is expected to be negatively related to default rate. | Flow |
| 3 | Household Debt | HHD | Positive | Debt affects borrower ability to repay loans. Higher household debt indicates that borrows are less likely to be able to make a repayment. | Flow |
| 4 | Corporate Debt | COPD | Negative | High corporate debt indicate good bond market which also reflect good economic. Hence, the relationship with default rate is negative. | Flow |
| 5 | Government Debt | GOVD | Positive | Government debt causes inflation increases and also default rate increases from inflation effects.  | Flow |
| 6 | Government Expenditure | GOVE | Negative | Government expenses increases, which indicates that the economy's growth increases. This will result in a negative projection between government expenses and default rate.  | Flow |
| 7 | Imports | IMP | Negative | Rising level of imports indicates robust domestic demand and a growing economy. The strengthening of economic activity is negatively related to the default rate. | Flow |
| 8 | Exports | EXP | Negative | Higher exports stimulate economic growth by increasing the aggregate demand of the economy. Hence, it will decrease default rate as the economy is in a good condition. | Flow |
| 9 | Policy Interest Rate  | PIR | Positive | The interest rate at which a depository institution lends funds to another depository institution (short-term) or the interest rate the central bank charges a financial institution to borrow money overnight. Overnight policy rates increases, lending cost increases, default rate increases. | Rate |
| 10 | Minimum Loan Rate | MLR | Positive | As the lending rate increases, the total cost of overall lending increases, leading to a increasing default rate. | Rate |
| 11 | Nominal Effective Exchange Rate Index | NEER | Positive | High NEER indicates a stronger currency which hurts exports and increases default rate. | Index |
| 12 | Real Effective Exchange Rate Index | REER | Positive | High REER indicates a stronger currency which hurts exports and increases default rate. | Index |
| 13 | Wage | WAGE | Negative | A higher wage indicates that the borrower will be more likely to pay the loan. Due to increased wealth which will then decrease default rate. | Price |
| 14 | Unemployment Rate | UNEM | Positive | Unemployment leads to forgone investment in economic growth as it indicates the cost of society for not fully running production. The relationship between business activities and unemployment rate is negative, hence the unemployment rate is assumed to be positively related to default rate. | Rate |
| 15 | Consumer Confidence Index | CCI | Negative | The increase in Consumer Confidence Index indicates an increase in degree of optimism on the state of the economy that consumers are expressing through their activities of spending and saving. In a better economy, we expect a decreasing default rate. | Index |
| 16 | Private Investment Index | PII | Negative | Higher investment indicates good economic performance, default rate should be decreased. | Index |
| 17 | Business Sentiment Index | BSI | Negative | A higher Leading Indicator indicates the economics growth and borrower will be more likely to pay off the debt due to this increased wealth which will then decrease default rate. | Index |
| 18 | Number of foreign tourists visiting Thailand | TOUR | Negative | A high number of tourists visiting Thailand indicates the economics growth and business will be more likely to gain income, which will then decrease default rate. | Flow |
| 19 | Oil Price | BROLP | Positive | Oil prices affect the prices of many consumer goods. A rising oil price increases the living cost of the borrowers. Therefore, due to reduced wealth, default is positively related to oil price as business cost get more expensive with the increase in oil price. | Price |
| 20 | Industrial Production Index | INDPRO | Negative | High Industrial Production Index indicates strong economic performance, default rate should be negatively decreased. | Index |
| 21 | Capacity Utilization Rate | CAPU | Negative | High Capacity Utilization Rate indicates strong economic performance, default rate should be negatively decreased. | Index |
| 22 | Broad Money | BROMO | Negative | High Broad money indicates strong economic performance, default rate should be negatively decreased. | Flow |
| 23 | Foreign Reserve | FRES | Negative | A large foreign reserve indicates strong economic performance, default rate should be negatively decreased. | Flow |
| 24 | Labour Index | LAB | Negative | High labour index indicates the stregthening labour market, with increased employment or wages. As a result, the default rate is expected to decreased. | Index |

</div>

The MEV Time series may not have a direct relationship with the dependent variable. Therefore, several transformations or alternative specifications may need to be considered to identify a meaningful relationship.

<div align="center">

| Transformation | Formula |
|---|---|
| Year-on-Year Changed (Rate) | $MEV_{t} - MEV_{t-12}$ |
| Year-on-Year Changed (Non-Rate) |  $(MEV_{t} - MEV_{t-12}) / MEV_{t-12}$ |
| Natural log transformation | $LN(MEV_{t})$ |
| Moving average | $(MEV_{t} + MEV_{t-1} + MEV_{t-n} + ...) / n$  |
| Leading indicator | $Lag_{t}(MEV)$ |

</div>

#### 2.3 Univariate Analysis
After completing the transformation of the MEV(s), preliminary assessments are conducted to further narrow down the candidate variables prior to multivariate analysis. An MEV is performed the single linear regression and retained if the variable meets the following criteria:
- p-value significant of 5% and;
- R-Square is higher than 50% and;
- It demonstrates an intuitive relationship with the dependent variable, with the expected direction of the relationship predefined.
> Note: MEV(s) that exhibiting either increasing or decreasing trends are permitted to proceed to subsequent analysis steps.

<p align="center">
<img width="989" height="590" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/bf8250bc-22dd-4893-a42c-54bce6b9af0e" />
</p>

```
[Univariate analysis]
=== Result ===
Number of passed variables: 214
```

#### 2.4 Multivariate Analysis
One of the commonly used industry methods to assess multicollinearity is variable clustering, which partitions a set of variables into non-overlapping clusters. This technique is implemented using the varclushi-opt library in Python. The objective is to form clusters in which variables are highly correlated with one another while exhibiting low correlation with variables in other clusters.
> The varclushi-opt can see further details on [varclushi_opt](https://github.com/naenumtou/varclushi_opt/tree/master).

An MEV is performed the cluster analysis and retained if the variable meets the following criteria:
- Top 2 lowest R-Square ratio per cluster
- Top 2 highest R-Square per cluster

<p align="center">
<img width="1965" height="789" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/e424229f-6a14-4468-b18c-fde20826d317" />
</p>

For model development purposes, all possible combinations of factors across clusters will be evaluated. This exhaustive assessment ensures that the full space of candidate models is explored, thereby increasing the likelihood of identifying the optimal model specification.
When generating factor combinations, the following constraints are imposed to control multicollinearity and preserve interpretability:
- A single combination must not include more than one variable from the same cluster.
- A single combination must not include more than one variable originating from the same pre-transformation group.

The number of variables included in each model may be adjusted based on empirical results or business considerations. However, the number of factors is capped at three, as including more than three variables typically increases the risk of multicollinearity and can adversely impact model stability and performance.

```
=== Result ===
Number of passed variables: 33

[Possible combinations of 1 variable(s)]
    Number of combinations: 33
[Possible combinations of 2 variable(s)]
    Number of combinations: 468
[Possible combinations of 3 variable(s)]
    Number of combinations: 3731

Totol combination for regression model: 4232
```

#### 2.5 Multiplie Linear Regression
Multiple linear regression is widely used in the industry for predictive modeling. This regression approach aims to estimate the relationship between a dependent variable and one or more independent variables, which in this context are macroeconomic variables. When a model includes two or more independent variables, it is referred to as a multiple linear regression model. The table below is summary of model diagnostic assumptions test for conclusion of BLUE (**B**est **L**inear **U**nbias **E**stimator):

<div align="center">
 
| Category | Description | Common tests | Risk (if not satisfied) | Passed criteria |
|:---:|---|---|---|:---:|
| p-value   significant | The p-value of   coefficients are less than or equal to 10% | OLS Regression | p-values is not   statistically significant | < 0.05 |
| Multicollinearity | Independent   variables are not strongly linearly related | Variance   inflation factor (VIF) | Imprecise   and/or ill-defined coefficient estimates | < 10 |
| Residual   normality | Residuals   follow a normal distribution | Anderson-Darling   test | Inaccurate   p-values | > 0.05 |
| Residual   homoscedasticity | Variance of   residuals is independent of the fitted value | White test | Inaccurate   p-values | > 0.05 |
| Residual   autocorrelation | Residuals are   not autocorrelated | Durbin Watson | Inaccurate   p-values; in particular, positive autocorrelation overstates significance of   variables | (1 - 3) |
| Residual   stationary | (Co-integration)   implying residuals are stationary, i.e. display constant mean and variance   over time | Augmented   Dickey-Fuller test | Inaccurate   p-values and misleading R-Square | < 0.1 |

</div>

#### 2.6 Model Back-testing

<p align="center">
<img width="989" height="593" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/3ebc0ae3-86bd-46c0-82f7-9d1fdaaa0a2b" />
</p>



### 3. Lifetime Model
<p align="center">
<img width="1647" height="955" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/0c231db8-8d81-408a-87b0-d1bf05a735c6" />
</p>

#### 3.1 Cohort Curve Construction
**Unbias Cohort Curves:** The cumulative PD curves derived from the unbias model is preparing for calibration. These must be transformed into weighted average conditional PD ready for lifetime calibration.
<p align="center">
<img width="989" height="590" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/8e8d2908-31c0-4ae6-b098-3ec66516b7ec" />
</p>



#### 3.2 Forward-looking Information
**PD Prediction:** The forward‑looking information is incorporated through outputs from the forward‑looking PD model. The macroeconomic variables is used as input for forecasting time‑varying PD, allowing historical cohort PDs to be adjusted to reflect expected future economic conditions across the projection horizon.

```
Final MEV(s): ['BROLP_MA9M_LAG9M', 'UNEM_MA9M_LAG3M', 'PIR_MA6M']
Forecasting PD
Year 1: 2.92%
Year 2: 1.45%
Year 3: 1.83%
```

#### 3.3 Calibration by Logit Approach
**Logit Approach:** The calibration using a logit approach to ensure numerical stability and proportional adjustment across time and segmentation. Cohort curves and the predicted PD(s) are transformed into logit space, where calibration factors are estimated to align adjusted PD curves with observed default outcomes while preserving smoothness, monotonicity, and lifetime plausibility.

The calibration process is done on logit space of conditional PD while the base cohort curves are computed by cumulative basis. The transformation is done by following formula to perform the calibration and convert back to final result.

$$
Marginal\ PD = Cumulative\ PD_{t + 1} - Cumulative\ PD_{t}
$$

$$
Conditional\ PD = \frac{{Marginal\ PD}_{t}}{1 - {Cumulative\ PD}_{t - 1}}
$$

$$
Calibration\ PD_{Portfolio} = Logit(Conditional\ PD_{t}) + Logit(FWL\ PD) - Logit(TTC\ PD)
$$

$$
Calibration\ PD_{Segment} = Logit(Conditional\ PD_{t}) + Logit(FWL\ PD) - Logit(TTC\ PD) + Delta
$$

where;

- $t$ is a time in lifetime period

#### 3.4 Optimization PiT PD
**Delta for PiT PD:** The optimization process is to minimize overall deviations between the adjusted average PD Curve of portfolio level and segment level PD Curves. In the other words, the forecasting PD from the forward-looking model should represent overall risk of the portfolio since the model has been done based on portfolio level. By seperating calibration by segment, it could cause the deviations from portfolio risk. This optimization process of delta is to ensures consistency across segments, avoids distortion of portfolio level risk, and results in optimized lifetime PD term structures that are forward‑looking, unbias, and suitable for IFRS 9 Reporting and risk management use.
<p align="center">
<img width="1964" height="1208" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ)" src="https://github.com/user-attachments/assets/cba773a0-df01-4ea9-8141-a94d41ef2551" />
</p>


