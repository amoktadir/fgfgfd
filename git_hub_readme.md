# Integrated MCDM Models (Streamlit App)

A Streamlit web application that implements **two Multi-Criteria Decision-Making (MCDM) models**:

1. **Trigonometric Trapezoidal Fuzzy OPA (TTrF-OPA)** — multi-expert criteria weighting using linguistic assessments aggregated with a trigonometric trapezoidal fuzzy weighted geometric operator, then solved via a fuzzy linear programming model.
2. **TTrFS-TRUST Method** — a **multi-normalization, multi-distance** assessment framework that supports **Soft (linguistic)** and **Hard (crisp)** criteria, expert aggregation, constraint-based normalization, and ranking of alternatives.

The app also supports **exporting results to Word (.docx)** for both models.

---

## ✨ Features

### ✅ Trigonometric Trapezoidal Fuzzy OPA (TTrF-OPA)
- Define **criteria** and **number of experts**
- Assign **expert weights** (validated to sum to 1)
- Experts provide **linguistic ratings** (ELI … EHI)
- Aggregates fuzzy importance via **TTrFWG** (trigonometric trapezoidal fuzzy weighted geometric)
- Builds **OPA coefficient set (θ)** and solves **fuzzy LP** using **PuLP**
- Displays:
  - aggregated fuzzy importance (θ)
  - coefficients
  - fuzzy weights (l, m, u, w)
  - ranked criteria
  - Ψ (psi) and defuzzified ψ
- **Export**: Word report of all key tables

### ✅ TTrFS-TRUST Method
- Step-by-step workflow with progress navigation:
  1. Problem Setup (alternatives, criteria, experts, α, β)
  2. Criteria Setup (Soft/Hard)
  3. Expert Weights
  4. Data Collection (soft linguistic + hard crisp)
  5. Decision Matrix (aggregation + defuzzification)
  6. Criteria Info (Benefit/Cost + weights)
  7. Constraint Intervals (ρᴸ, ρᵁ)
  8. Results (multi-normalization + multi-distance ranking)
- Supports **four normalization schemes**:
  - Linear ratio-based (r)
  - Linear sum-based (s)
  - Max–Min (m)
  - Logarithmic (l)
- Computes distance measures:
  - Euclidean (ε)
  - Manhattan (π)
  - Lorentzian (ℓ)
  - Pearson (ρ)
- Produces final score **ℒ** and ranking
- **Export**: Word report including decision matrix, normalization matrices, and final ranking

---

## 🧰 Tech Stack

- **Python**
- **Streamlit** (UI)
- **NumPy / Pandas** (data & computation)
- **PuLP** (linear programming)
- **python-docx** (Word report generation)

---

## 📦 Project Structure (Recommended)

```text
.
├─ app.py                 # main Streamlit app (your code)
├─ requirements.txt
├─ README.md
└─ assets/                # optional screenshots / demo gifs
```

> If your file name is different (e.g., `main.py`), update commands below accordingly.

---

## 🚀 Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2) Create and activate a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the Streamlit app

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`). Open it in your browser.

---

## 🧾 requirements.txt

Create a `requirements.txt` file with the following (versions optional):

```txt
streamlit
numpy
pandas
pulp
python-docx
```

> If you deploy to Streamlit Community Cloud, keep dependencies minimal and consistent.

---

## 🧪 Usage Guide

### A) TTrF-OPA Model (Criteria Weighting)
1. Select **“Trigonometric Trapezoidal Fuzzy OPA”** from the sidebar.
2. Enter:
   - Number of experts
   - Number of criteria
   - Criterion names
3. Set **expert weights** (must sum to **1.00**).
4. For each expert, choose a linguistic rating for each criterion:
   - `ELI, VLI, LI, MI, HI, VHI, EHI`
5. Click **Calculate Weights**.
6. Review:
   - aggregated θ
   - coefficients
   - fuzzy weights and rank
   - ψ values
7. Click **Export Results to Word** to download a report.

### B) TTrFS-TRUST Method (Ranking Alternatives)
1. Select **“TTrFS-TRUST Method”** from the sidebar.
2. Follow the steps in order:
   - **Problem Setup:** set α (must sum to 1) and β
   - **Criteria Setup:** mark each criterion as Soft/Hard
   - **Expert Weights:** must sum to 1
   - **Data Collection:**
     - Soft criteria → linguistic ratings by each expert
     - Hard criteria → crisp numeric values
   - **Decision Matrix:** aggregated and defuzzified matrix
   - **Criteria Information:** Benefit/Cost + criterion weights (sum to 1)
   - **Constraints:** enter ρᴸ and ρᵁ (bounds)
   - **Results:** view normalization matrices, distances, and final ℒ ranking
3. Export the report using **Export TRUST Results to Word**.

---

## 🧠 Method Notes (Quick)

### Linguistic Terms
- OPA model uses:
  - `ELI, VLI, LI, MI, HI, VHI, EHI`
- TRUST model uses:
  - `ELI, VLI, LI, MLI, MI, MHI, HI, VHI, EHI`

### Key Parameters
- **α = (∂₁, ∂₂, ∂₃, ∂₄)**: weights for the four normalization methods (should sum to 1)
- **β**: distance aggregation parameter (0 to 1)

---

## 🖼️ Screenshots (Optional)

Add screenshots to `assets/` and reference them here:

```md
![OPA Model](assets/opa.png)
![TRUST Model](assets/trust.png)
```

---

## ☁️ Deployment

### Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud and select the repository.
3. Set:
   - **Main file path:** `app.py`
4. Ensure `requirements.txt` is present.

---

## 🛠️ Troubleshooting

### PuLP solver issues
- If optimization fails or solver is missing, ensure PuLP is installed:

```bash
pip install pulp
```

- Some environments require an external solver, but PuLP often works out-of-the-box with default CBC.

### Expert weights validation
- The app requires expert weights to sum to **1.00**.
- If you see a validation error, adjust weights until the displayed sum is exactly **1.00**.

### Streamlit rerun behavior
- The app uses `st.rerun()` to refresh state after edits.
- If you see frequent reruns, try editing a full row then clicking outside the table.

---

## 📄 Citation

If you use this tool in academic work, please cite the associated paper/methodology (add your paper reference here):

```bibtex
@article{yourkey,
  title={...},
  author={...},
  journal={...},
  year={...}
}
```

---

## 👤 Author

Developed by **AAA** (update with your name/affiliation).

---

## 📜 License

Choose a license and add a `LICENSE` file (e.g., MIT, Apache-2.0).

Example (MIT):
- Add a `LICENSE` file with the MIT text
- Keep this README section as-is

