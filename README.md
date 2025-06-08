
# ☀️ Solar Panel Efficiency Prediction using Machine Learning

Welcome to the Solar Panel Efficiency Prediction project! This repository showcases a machine learning pipeline to forecast the performance and efficiency of solar panels using real-world environmental and sensor data.

## 📌 Project Objective

The goal of this project is to **predict the efficiency of solar panels** based on various parameters such as temperature, irradiance, humidity, panel age, and more. This helps in identifying performance degradation and potential failures, leading to smarter maintenance and optimized energy yield.

---

## 🧠 Models Used

| Model               | Accuracy (%) |
|--------------------|--------------|
| **Linear Regression** | **89.37**    |
| Gradient Boosting  | 89.11        |
| Random Forest      | 88.78        |
| XGBoost            | 88.33        |

🔍 **Best Model**: *Linear Regression* (R² score = **0.8937**)

---

## 📁 Dataset Overview

> **Source**: Zelekstra ML Challenge Dataset  
> [Reference Paper on SSRN](https://ssrn.com/abstract=4771205)

### 🔧 Features Used:

| Column Name        | Description |
|--------------------|-------------|
| `temperature`        | Ambient air temperature (°C) |
| `irradiance`         | Solar irradiance (W/m²) |
| `humidity`           | Relative humidity (%) |
| `panel_age`          | Age of the panel (years) |
| `maintenance_count`  | Number of maintenance events |
| `soiling_ratio`      | Dirt/debris impact ratio (0–1) |
| `voltage`            | Output voltage (V) |
| `current`            | Output current (A) |
| `module_temperature` | Panel surface temperature |
| `cloud_coverage`     | Cloud coverage (%) |
| `wind_speed`         | Wind speed (m/s) |
| `pressure`           | Atmospheric pressure (hPa) |
| `string_id`          | Identifier for panel grouping |
| `error_code`         | Diagnostic error code |
| `installation_type`  | Type of installation (fixed, tracking, dual-axis) |
| `efficiency`         | **Target variable** - actual panel efficiency |

---

## 📊 Exploratory Data Analysis

- Missing values were handled using imputation techniques.
- Categorical variables like `error_code`, `string_id`, and `installation_type` were label-encoded.
- Feature importance analysis was performed to understand the key drivers of efficiency.

---

## 🛠️ Tech Stack

- **Language**: Python  
- **Libraries**:
  - `pandas`, `numpy` for data manipulation
  - `scikit-learn` for modeling
  - `matplotlib`, `seaborn` for visualization
  - `xgboost`, `lightgbm` for gradient boosting models

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/tanishqkolhatkar93/Solar_Panel_Efficiency_Prediction.git
   cd Solar_Panel_Efficiency_Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   - Open `Solar_Panel_Efficiency_Prediction.ipynb` in Jupyter or VS Code.
   - Run all cells to view the full pipeline from preprocessing to model evaluation.

---

## 📈 Results & Insights

## 📊 Model Results Visualization

![Efficiency Plot](assets/efficiency_plot.png)
*Efficiency of solar panel over varying environmental and sensor data.*


---

## 🎥 Project Demo

![Demo](assets/demo.gif)
*Quick overview of model training and prediction process.*


- **Linear Regression** performed the best, proving that even a simple model can be effective when features are well-engineered.
- `irradiance`, `temperature`, and `soiling_ratio` were the top contributors to efficiency variation.
- The solution can help predict solar panel degradation in smart energy systems.

---

## 📚 References

- 📄 [SSRN Paper – Solar Panel Performance Prediction](https://ssrn.com/abstract=4771205)
- 🔗 Zelekstra ML Challenge Dataset

---

## 👨‍💻 Author

**Tanishq Kolhatkar**  
💼 [LinkedIn](https://www.linkedin.com/in/tanishqkolhatkar) • 💡 AI/ML & Cloud Enthusiast

---

## ⭐ Show Your Support

If you found this useful, consider giving a ⭐️ to the repo and sharing it with your network!
