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


fdsafdsa


#### 1.1 Cohort Building
**Cohort Based Default Measurement:** Exposures are segmented into homogeneous cohorts defined at observation point (e.g., vintage, product type, risk band). For each cohort, default events are tracked over time since observation point to construct marginal and cumulative default "triangles" forming the empirical basis for lifetime PD estimation under an IFRS 9 consistent default definition.

#### 1.2 Chain-Ladder
**Chain‑Ladder Pattern Projection:** Observed cohort default triangles are now run-off triangles. Given this fact that the more recent period, the less lifetime windows for tracking default. To fill the triangles, using chain‑ladder (development factor) methodology to extrapolate incomplete default histories. This produces projected ultimate cumulative lifetime PD for each cohort, enabling consistent estimation even for recently originated (immature) portfolios.

#### 1.3 Gamma Fitting
**Parametric Model via Gamma Distribution:** A Gamma distribution is fitted to the projected cumulative PD term structure for each cohort to remove sampling noise, enforce monotonicity, and obtain a smooth, stable PD curve. Differentiation of the fitted cumulative curve yields period‑specific (marginal) PDs with appropriate skewness and long‑term behavior.

#### 1.4 Unbias Calibration
#### 1.5 Result

### 2. Forward-looking Model
<p align="center">
<img width="1690" height="931" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/fe580c69-d4d4-4728-99f2-ee073f5d786f" />
</p>


### 3. Lifetime Model
<p align="center">
<img width="1647" height="955" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/0c231db8-8d81-408a-87b0-d1bf05a735c6" />
</p>
