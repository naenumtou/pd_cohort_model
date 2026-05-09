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
|   |   └── mev_data.csv
├── requirements.txt
└── README.md
```

## Project Details
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
<p align="center">
<img width="1990" height="789" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/a15caf28-afe1-4d51-b946-c9a865c426d6" />
</p>

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

#### 2.1 Macroeconomics Variables Transformation
#### 2.2 Univariate Analysis

<p align="center">
<img width="989" height="590" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/bf8250bc-22dd-4893-a42c-54bce6b9af0e" />
</p>


```
[Univariate analysis]
=== Result ===
Number of passed variables: 214
```

#### 2.3 Multivariate Analysis

<p align="center">
<img width="1965" height="789" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/e424229f-6a14-4468-b18c-fde20826d317" />
</p>

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


#### 2.4 Multiplie Linear Regression


#### 2.5 Model Back-testing

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



#### 3.2 Forward-looking Infornation
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





