# ⚡ AI-Predictor for Hydrogen Production for SCWG

An advanced, machine-learning-powered web application built with **Streamlit** to predict hydrogen yield from **Supercritical Water Gasification (SCWG)**. This tool provides researchers and engineers with a quick way to estimate hydrogen production and its associated environmental benefits based on waste composition and process parameters.



## 🚀 Overview

Supercritical Water Gasification (SCWG) is a sustainable technology that converts high-moisture biomass and waste into hydrogen-rich syngas. This application uses a **Random Forest Regression** model to predict the output based on experimental variables.

### Key Metrics Tracked:
* **Hydrogen Yield:** Measured in mol/kg of waste.
* **CO2 Reduction:** Calculated in $kgCO_2e$ based on waste type.
* **Environmental Equivalence:** Visualizes impact through "Tree-Years" and car travel distance saved.

---

## 📁 Repository Structure

```text
.
├── models/
│   ├── rf_model.pkl         # Trained Random Forest Model
│   └── scaler.pkl           # StandardScaler for input normalization
├── app.py                   # Main Streamlit application file
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
🛠️ Installation & SetupTo run this project locally, follow these steps:1. Clone the RepositoryBashgit clone [https://github.com/yourusername/hydrogen-scwg-predictor.git](https://github.com/yourusername/hydrogen-scwg-predictor.git)
cd hydrogen-scwg-predictor
2. Set Up a Virtual EnvironmentBashpython -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate
3. Install RequirementsBashpip install -r requirements.txt
4. Run the ApplicationBashstreamlit run app.py
🧪 Model InformationThe underlying AI model analyzes eight specific input features:Carbon (C) %Hydrogen (H) %Nitrogen (N) %Oxygen (O) %Solid Content %Temperature (°C) (Optimized for 300-650°C)Pressure (MPa) (Optimized for 10-35 MPa)Reaction Time (min)🌍 Environmental Impact FactorsThe app utilizes industry-standard factors to provide real-world context to the data:Carbon Sequestration: 25 kg $CO_2$ per tree per year.Car Emissions: 0.250 kg $CO_2$ per km (based on a 1.8L gasoline engine).Blue H2 Savings: 1.6 kg $CO_2$ saved per kg of $H_2$ produced compared to traditional gasoline.📦 DependenciesStreamlit: For the web interface.Scikit-learn: For model loading and scaling.Numpy: For numerical processing.Pickle: For model serialization.📄 LicenseThis project is licensed under the MIT License.
---

### Next Step
To make this repository truly "plug-and-play," I can also generate the **requirements.txt**
