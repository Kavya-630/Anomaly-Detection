##  Dataset

The dataset used in this project is:

**Cold Rolling Mill: Dataset for Recommender System for Process Optimization**

It can be downloaded from:

🔗 [Zenodo – DOI: 10.5281/zenodo.8085933](https://doi.org/10.5281/zenodo.8085933)

Note: The dataset file (`dataset.pkl`) must be downloaded manually and placed inside the `data/` folder.

## Dataset Overview
- **Total Records:** 311,542  
- **Total Features:** 60  
- **Data Type:** Fully Numerical  
- **Memory Usage:** ~142.6 MB  
- **Missing Values:** None (all 311,542 entries are non-null)

##  Identification of Equipment / Process Parameters
The dataset contains variables related to cold rolling mill operations. The features can be categorized as follows:
### A. Coil & Process Information
---
- `coil_id` → Unique steel coil identifier  
- `pass_nr` → Rolling pass number  

These parameters help track rolling stages and production batches.

---

### B. Thickness Control Parameters

- `REF_INITIAL_THICKNESS`  
- `REF_TARGET_THICKNESS`  
- `h_entry_ref`  
- `h_exit_ref`  
- `EXIT_THICK_DEVIATION_ABS_AVG`  
- `EXIT_THICK_DEVIATION_AVG`  

These are critical CRM (Cold Rolling Mill) quality indicators and are highly relevant for industrial process monitoring and optimization.

---

### C. Velocity & Speed Parameters

- `velocity_mdr`  
- `velocity_en`  
- `velocity_ex`  
- `velocity_en_max`, `velocity_en_min`, `velocity_en_std`  
- `velocity_ex_max`, `velocity_ex_min`, `velocity_ex_std`  

These parameters represent strip movement and rolling dynamics within the mill.

---

### D. Tension & Force Parameters

- `tension_en`  
- `tension_ex`  
- `rgc_act`  
- `dh_entry`, `dh_entry_std`, etc.  
- `dm_bur`, `dm_imr1`, `dm_wrbot`, `dm_wrtop`  

These variables are related to mechanical deformation and roll gap control, and are strongly linked to equipment performance and operational stability.

---

### E. Chemical Composition Parameters

The dataset includes alloy composition elements such as:

`C, SI, MN, S, P, CR, NI, CU, AL, SN, MO, V, TI, NB, W, N, CO, ZR, B, TE, PB, SB, CA, TA, CE, LA`

These chemical properties influence rolling behavior, deformation characteristics, and final thickness quality.

---

## class Variable

- `class` → Type: `int64`

**Interpretation:**

- `0` → Category A 
- `1` → Category B
-  

This makes the problem a **Binary Classification Task**.

---

##  Suitable Machine Learning Models

Based on the dataset structure, the following models are appropriate:

- Logistic Regression  
- Random Forest  
- XGBoost  

These models can be used to classify rolling mill process conditions and detect abnormal operational states.
