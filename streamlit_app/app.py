"""
Neonatal ARDS research classifier - Streamlit web app.

Loads the locked random-forest model (ards_rf_model.joblib) exported by
train_export_model.py and provides an interactive risk calculator with
individual-level SHAP explanation.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "ards_rf_model.joblib"
META_PATH = HERE / "model_metadata.json"

FEATURE_LABELS = {
    "AI": "Antenatal infection",
    "ACS": "Antenatal corticosteroids",
    "MA": "Maternal age (years)",
    "GH": "Gestational hypothyroidism",
    "GDM": "Gestational diabetes mellitus",
    "SII": "Systemic immune-inflammation index",
}
FEATURE_HELP = {
    "AI": "Antenatal infection documented within 14 days before delivery (No / Yes).",
    "ACS": "Any antenatal corticosteroid administration (No / Yes).",
    "MA": "Maternal age at delivery, in years.",
    "GH": "Diagnosis of hypothyroidism during pregnancy (No / Yes).",
    "GDM": "Diagnosis of gestational diabetes mellitus (No / Yes).",
    "SII": "SII = Platelet count x Neutrophil count / Lymphocyte count.",
}

EXPECTED_FEATURES = ["AI", "ACS", "MA", "GH", "GDM", "SII"]

CUSTOM_CSS = """
<style>
/* ---- Global palette & typography ---- */
html, body, [class*="css"]  { font-family: "Inter", "Segoe UI", system-ui, sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1150px; }

h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; color: #12233f; letter-spacing:.2px; }

/* ---- Hero banner ---- */
.hero {
  background: linear-gradient(135deg, #12233f 0%, #1f3a5f 55%, #0f766e 140%);
  color: #f3f6fb; padding: 1.6rem 1.9rem; border-radius: 16px;
  box-shadow: 0 10px 30px rgba(18,35,63,.18); margin-bottom: 1.4rem;
}
.hero h1 { color:#ffffff; margin:0 0 .35rem 0; font-size: 1.9rem; }
.hero p  { color:#cbd8ea; margin:0; font-size: .98rem; }
.hero .eyebrow {
  text-transform: uppercase; letter-spacing: 2.5px; font-size:.72rem;
  color:#8fb8d6; font-weight:600; margin-bottom:.5rem;
}

/* ---- Cards ---- */
.card {
  background:#ffffff; border:1px solid #e6eaf0; border-radius:14px;
  padding:1.3rem 1.4rem; box-shadow:0 2px 10px rgba(18,35,63,.05);
}
.section-title {
  font-family: Georgia, serif; font-size:1.12rem; color:#12233f;
  border-left:4px solid #0f766e; padding-left:.6rem; margin:.1rem 0 1rem 0;
}

/* ---- Result panel ---- */
.result {
  border-radius:14px; padding:1.5rem 1.6rem; border:1px solid;
}
.result .prob { font-size:3rem; font-weight:700; line-height:1; font-family:Georgia,serif; }
.result .band { font-size:1.15rem; font-weight:600; margin-top:.2rem; }
.result .cls  { color:#3d4a5c; margin-top:.55rem; font-size:.95rem; }

/* ---- Metric chips in sidebar ---- */
.stMetric { background:#f6f8fb; border-radius:10px; padding:.5rem .7rem; }

/* ---- Buttons ---- */
.stButton>button, .stFormSubmitButton>button {
  background:#0f766e; color:#fff; border:none; border-radius:10px;
  font-weight:600; padding:.55rem 1rem;
}
.stButton>button:hover, .stFormSubmitButton>button:hover { background:#0b5c55; color:#fff; }

/* ---- Footer note ---- */
.disclaimer { color:#6b7688; font-size:.82rem; border-top:1px solid #e6eaf0;
  margin-top:1.8rem; padding-top:.9rem; }
</style>
"""


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    # A single interactive prediction does not benefit from worker-pool setup,
    # and one thread keeps the calculator compatible with restricted hosts.
    model.named_steps["model"].n_jobs = 1
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


@st.cache_resource
def get_shap_explainer(_model):
    """Build a TreeExplainer on the fitted random-forest step."""
    try:
        import shap

        rf = _model.named_steps["model"]
        pre = _model.named_steps["preprocess"]
        explainer = shap.TreeExplainer(rf)
        return explainer, pre
    except Exception:
        return None, None


def classification_display(prob: float, threshold: float) -> tuple[str, str]:
    """Return the model classification without inventing unvalidated risk bands."""
    if prob >= threshold:
        return "Above the reporting threshold", "#b42318"
    return "Below the reporting threshold", "#175cd3"


def main() -> None:
    st.set_page_config(
        page_title="Neonatal ARDS Research Classifier",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if not MODEL_PATH.exists() or not META_PATH.exists():
        st.error(
            "Model artifacts not found. Run `python train_export_model.py` first "
            "to generate ards_rf_model.joblib and model_metadata.json."
        )
        st.stop()

    model, meta = load_artifacts()
    features = meta["features"]
    if features != EXPECTED_FEATURES:
        st.error(
            "The deployed artifact does not match the manuscript-locked six-predictor "
            "model. Expected AI, ACS, MA, GH, GDM and SII."
        )
        st.stop()
    binary_features = set(meta["binary_features"])
    threshold = float(meta["primary_threshold"])
    ref = meta["feature_reference"]

    # ------------------------------------------------------------------ Sidebar
    with st.sidebar:
        st.markdown("### Model card")
        st.markdown(
            f"""
**Algorithm** &nbsp; Random forest  
**Feature set** &nbsp; {meta['feature_set']}  
**Predictors** &nbsp; {len(features)}  
**Operating threshold** &nbsp; Youden = **{threshold:.3f}**  
**Development / test** &nbsp; n = {meta['n_train']} / {meta['n_test']}
"""
        )
        tm = meta["test_metrics_primary_threshold"]
        st.markdown("#### Internal test-set performance")
        c1, c2 = st.columns(2)
        c1.metric("AUC", f"{tm['auc']:.3f}")
        c2.metric("AP", f"{tm['pr_auc']:.3f}")
        c3, c4 = st.columns(2)
        c3.metric("Sensitivity", f"{tm['sensitivity']:.2f}")
        c4.metric("Specificity", f"{tm['specificity']:.2f}")
        st.caption(
            "Research prototype evaluated in an internal test set from the same "
            "cohort; not externally validated. The displayed probability is not "
            "a transportable absolute risk estimate."
        )

    # ------------------------------------------------------------------- Header
    st.markdown(
        """
<div class="hero">
  <div class="eyebrow">Research-use classification model</div>
  <h1>Neonatal ARDS Research Classifier</h1>
  <p>For retrospective research in a cohort of mechanically ventilated
     late-preterm and term neonates with respiratory distress, this interface
     distinguishes neonatal ARDS from pooled non-ARDS respiratory conditions
     using six maternal predictors available immediately before delivery.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.warning(
        "Target population and timing: this model was developed only in neonates "
        "with respiratory distress requiring mechanical ventilation. It was not "
        "evaluated for antenatal screening of unselected pregnancies, cannot be "
        "used before target-population eligibility is known, and must not guide "
        "the initial decision to administer antenatal corticosteroids."
    )

    left, right = st.columns([1, 1.15], gap="large")

    # --------------------------------------------------------------- Input form
    with left:
        st.markdown('<div class="section-title">Patient inputs</div>', unsafe_allow_html=True)
        with st.form("predict_form"):
            values = {}
            for f in features:
                label = FEATURE_LABELS.get(f, f)
                help_txt = FEATURE_HELP.get(f, "")
                if f in binary_features:
                    choice = st.radio(
                        label, ["No", "Yes"], horizontal=True, help=help_txt
                    )
                    values[f] = 1 if choice == "Yes" else 0
                elif f == "MA":
                    values[f] = st.number_input(
                        label,
                        min_value=15.0,
                        max_value=55.0,
                        value=float(ref[f]["median"]),
                        step=1.0,
                        help=help_txt,
                    )
                elif f == "SII":
                    st.markdown("**Systemic immune-inflammation index (SII)**")
                    mode = st.radio(
                        "SII input mode",
                        ["Enter SII directly", "Compute from PLT / NEU / LYM"],
                        help=FEATURE_HELP["SII"],
                    )
                    if mode == "Enter SII directly":
                        values[f] = st.number_input(
                            "SII value",
                            min_value=0.0,
                            max_value=20000.0,
                            value=float(ref[f]["median"]),
                            step=10.0,
                        )
                    else:
                        plt_v = st.number_input("Platelet count (PLT, 10^9/L)", 1.0, 1000.0, 200.0, 1.0)
                        neu_v = st.number_input("Neutrophil count (NEU, 10^9/L)", 0.1, 50.0, 6.0, 0.1)
                        lym_v = st.number_input("Lymphocyte count (LYM, 10^9/L)", 0.1, 20.0, 1.8, 0.1)
                        sii_calc = plt_v * neu_v / lym_v if lym_v > 0 else 0.0
                        st.info(f"Computed SII = {sii_calc:,.1f}")
                        values[f] = sii_calc
                else:
                    values[f] = st.number_input(
                        label, value=float(ref[f]["median"]), help=help_txt
                    )
            submitted = st.form_submit_button(
                "Calculate model estimate", type="primary", use_container_width=True
            )

    # ---------------------------------------------------------------- Prediction
    with right:
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
        if not submitted:
            st.info(
                "Complete the inputs on the left and click "
                "**Calculate model estimate**."
            )
            _render_disclaimer()
            return

        X = pd.DataFrame([{f: values[f] for f in features}])
        prob = float(model.predict_proba(X)[:, 1][0])
        band, color = classification_display(prob, threshold)
        predicted_class = "ARDS-outcome positive" if prob >= threshold else "ARDS-outcome negative"

        st.markdown(
            f"""
<div class="result" style="background:{color}12;border-color:{color};">
  <div class="prob" style="color:{color};">{prob:.3f}</div>
  <div class="band" style="color:{color};">{band}</div>
  <div class="cls">Model score and classification at the training-derived
     Youden reporting threshold ({threshold:.3f}): <b>{predicted_class}</b></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            "The numeric output is a model-estimated probability in this selected "
            "development setting. It is conditional on the cohort case mix (ARDS "
            "prevalence 47.1% overall and 47.8% in the internal test set) and is "
            "not a calibrated absolute risk for other populations."
        )

        _render_shap(model, X)

        with st.expander("Show model inputs used"):
            disp = X.T.rename(columns={0: "value"})
            st.dataframe(disp, use_container_width=True)

        _render_disclaimer()


def _render_shap(model, X: pd.DataFrame) -> None:
    explainer, pre = get_shap_explainer(model)
    if explainer is None:
        return
    try:
        import matplotlib.pyplot as plt

        X_proc = pre.transform(X)
        proc_names = list(pre.get_feature_names_out())
        sv = explainer.shap_values(X_proc)
        if isinstance(sv, list):
            sv_pos = sv[1][0]
        else:
            arr = np.asarray(sv)
            sv_pos = arr[0, :, 1] if arr.ndim == 3 else arr[0]

        st.markdown("#### Why this prediction? (SHAP contributions)")
        contrib = (
            pd.DataFrame({"feature": proc_names, "shap_value": sv_pos})
            .assign(abs_v=lambda d: d["shap_value"].abs())
            .sort_values("abs_v", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        colors = ["#b42318" if v > 0 else "#175cd3" for v in contrib["shap_value"]]
        ax.barh(contrib["feature"], contrib["shap_value"], color=colors)
        ax.axvline(0, color="#12233f", lw=0.8)
        ax.set_xlabel("SHAP value (rightward = higher risk)")
        ax.invert_yaxis()
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Red bars push the model output toward ARDS classification; "
            "blue bars push it toward non-ARDS classification."
        )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"SHAP explanation unavailable: {exc}")


def _render_disclaimer() -> None:
    st.markdown(
        """
<div class="disclaimer">
Research prototype. Predictions are model estimates based on a single-cohort,
internally evaluated model. The model has uncertain calibration, lacks external
validation and prospective clinical-impact evaluation, and must not be used for
screening or as the basis for any clinical decision.
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
