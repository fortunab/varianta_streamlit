import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit settings
st.set_page_config(page_title="Federated Learning Model Comparison", layout="wide")
st.title("📊 Federated Learning Accuracy Comparison by Use Case")

# Simulated Data
rounds = list(range(1, 11))


# CP 1 dataset, CC 3 datasets, Alz 2 datasets

# Simulated results for each use case and model
simulated_data = {
    "Colorectal Polyps": {
        "LLM":      [72.5, 76.8, 79.4, 82.1, 85.3, 88.6, 90.2, 92.4, 94.0, 95.8],
        "ViT":      [70.2, 74.5, 77.1, 80.3, 83.9, 86.1, 88.5, 90.7, 92.5, 94.1],
        "Autoencoder":      [68.1, 71.0, 73.6, 76.2, 78.8, 81.0, 83.1, 85.3, 87.6, 89.7],
        "CNN": [65.0, 68.5, 70.2, 72.4, 74.6, 76.8, 78.0, 80.0, 81.5, 83.2],
        "CNN + RF": [63.2, 66.7, 68.4, 70.0, 71.9, 73.5, 75.0, 76.3, 77.9, 79.1]
    },
    "Cervical Cancer": {
        "LLM":      [70.1, 73.5, 76.0, 78.4, 81.7, 84.0, 86.3, 88.1, 90.2, 91.8],
        "ViT":      [68.0, 71.2, 74.0, 76.7, 79.5, 81.9, 84.0, 86.0, 87.9, 89.3],
        "Autoencoder":      [65.5, 68.3, 70.9, 73.5, 75.8, 77.9, 79.8, 81.5, 83.4, 85.0],
        "CNN": [62.0, 64.8, 67.2, 69.0, 71.1, 73.2, 74.9, 76.0, 77.6, 78.5],
        "CNN + RF": [60.5, 63.2, 65.1, 66.9, 69.0, 70.6, 72.3, 73.8, 75.0, 76.2]
    },
    "Alzheimer’s Disease": {
        "LLM":      [69.8, 72.7, 75.1, 78.2, 81.5, 84.1, 86.2, 88.3, 90.1, 91.9],
        "ViT":      [67.5, 70.3, 73.0, 75.9, 78.7, 81.1, 83.4, 85.3, 87.2, 89.0],
        "Autoencoder":      [65.0, 67.7, 70.2, 72.8, 75.0, 77.1, 78.9, 80.6, 82.7, 84.3],
        "CNN": [61.2, 63.9, 66.1, 68.5, 70.4, 72.3, 74.1, 75.6, 77.0, 78.4],
        "CNN + RF": [59.8, 62.1, 64.2, 66.3, 68.2, 69.9, 71.4, 72.7, 74.1, 75.3]
    }
}

# ---------- Sidebar Controls ----------
use_cases = list(simulated_data.keys())
selected_use_case = st.sidebar.selectbox("🧬 Select Clinical Use Case", use_cases)

available_models = list(simulated_data[selected_use_case].keys())
selected_models = st.sidebar.multiselect(
    "🧠 Select Models to Compare", available_models, default=available_models
)

# ---------- Data Preparation ----------
df = pd.DataFrame({
    "Round": rounds
})

for model in selected_models:
    df[model] = simulated_data[selected_use_case][model]

df_melted = df.melt(id_vars="Round", var_name="Model", value_name="Accuracy")

# ---------- Charts ----------
st.subheader(f"📈 Accuracy over Rounds – {selected_use_case}")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df_melted, x="Round", y="Accuracy", hue="Model", marker="o", ax=ax)
ax.set_title(f"Model Accuracy Comparison for {selected_use_case}")
ax.set_ylim(55, 100)
st.pyplot(fig)

# Table
st.subheader("📋 Accuracy Table")
st.dataframe(df.set_index("Round"))


# ---------------------- XAI Section ---------------------- #
import numpy as np

st.markdown("---")
st.subheader("🧠 Explainable AI (XAI) – Feature Importance")

# Simulated SHAP-style feature importances
features = [f"Feature {i+1}" for i in range(8)]
importances = np.round(np.random.dirichlet(np.ones(8), size=1)[0] * 100, 2)

df_xai = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values("Importance", ascending=True)

fig_xai, ax_xai = plt.subplots()
ax_xai.barh(df_xai["Feature"], df_xai["Importance"], color="teal")
ax_xai.set_title("Simulated Feature Importance (SHAP-style)")
ax_xai.set_xlabel("Relative Importance (%)")
st.pyplot(fig_xai)

# Textual LLM-style summary
st.subheader("🗣️ Model Explanation Summary")
explanation = f"""
Based on the analysis for **{selected_use_case}**, using **{', '.join(selected_models)}**, 
the model emphasizes diagnostic relevance in **Feature 3** and **Feature 7**.  
Attention maps suggest the model focuses on input regions or patterns strongly associated with clinical markers,  
supporting a transparent decision-making process in the federated setup.
"""
st.markdown(explanation)

# Attention heatmap
st.subheader("🛰️ Simulated Attention Map")
attention_matrix = np.random.rand(10, 10)
fig_att, ax_att = plt.subplots()
sns.heatmap(attention_matrix, cmap="YlGnBu", ax=ax_att)
ax_att.set_title("Simulated Attention Heatmap")
st.pyplot(fig_att)

# ---------------------- Zero-Shot Proxy Comparison ---------------------- #
import numpy as np

# ---------------------- Zero-Shot + NAS Integration Results ---------------------- #
st.markdown("---")
st.subheader("🚀 Zero-Shot + NAS Integration Results")

# Get model names from current selection
model_names = selected_models

# Calculate final-round accuracy for each selected model
baseline_scores = []
for model in model_names:
    try:
        baseline_scores.append(df[model].iloc[-1])
    except KeyError:
        baseline_scores.append(np.nan)

# Simulate NAS + zero-shot boosted scores (capped at 97%)
nas_proxy_scores = [
    min(97.0, score + np.random.uniform(1.5, 3.5)) if not np.isnan(score) else np.nan
    for score in baseline_scores
]

# Build DataFrame
df_nas_proxy = pd.DataFrame({
    "Model": model_names,
    "Baseline Accuracy (%)": np.round(baseline_scores, 2),
    "With NAS + Zero-Shot (%)": np.round(nas_proxy_scores, 2)
})
df_nas_proxy["Δ Accuracy (%)"] = df_nas_proxy["With NAS + Zero-Shot (%)"] - df_nas_proxy["Baseline Accuracy (%)"]
df_nas_proxy = df_nas_proxy.round(2)

# Display table
st.dataframe(df_nas_proxy.set_index("Model"), use_container_width=True)

# Optional bar chart
st.subheader("📈 Accuracy Gains with NAS + Zero-Shot")
fig_nas, ax_nas = plt.subplots()
sns.barplot(data=df_nas_proxy, x="Model", y="Δ Accuracy (%)", palette="viridis", ax=ax_nas)
ax_nas.set_ylim(0, df_nas_proxy["Δ Accuracy (%)"].max() + 1.5)
ax_nas.set_title("Model Improvements with NAS and Zero-Shot Integration")
st.pyplot(fig_nas)

