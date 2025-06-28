# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from src.feature_engineer import FeatureEngineer
import shap

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy Consumption Forecaster",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Load Artifacts ---
# Use @st.cache_resource to load these heavy objects only once.
@st.cache_resource
def load_artifacts():
    """Loads all necessary models, scalers, data, and feature lists."""
    try:
        model_median = joblib.load("models/model_Median_Forecast (50th).pkl")
        model_peak = joblib.load("models/model_Peak_Forecast (90th).pkl")
        scaler_ar = joblib.load("models/scaler_ar.pkl")
        scaler_env = joblib.load("models/scaler_env.pkl")
        full_data = pd.read_csv("data/energydata_complete.csv", parse_dates=["date"])
    except FileNotFoundError:
        st.error("Model/scalers/data not found. Please run `python train.py` first.")
        return [None] * 5

    if "rv1" in full_data.columns and "rv2" in full_data.columns:
        full_data = full_data.drop(columns=["rv1", "rv2"])

    # Create the feature list from a sample to ensure consistency
    sample_df = full_data.head(200).copy()
    fe = FeatureEngineer(sample_df.set_index("date"))
    df_engineered = (
        fe.create_time_features()
        .create_cyclical_features()
        .create_lag_and_rolling_features()
        .df
    )
    features = [
        col for col in df_engineered.columns if col != "Appliances" and col != "lights"
    ]

    return model_median, model_peak, scaler_ar, scaler_env, full_data, features


def get_shap_explainer(_model):
    """Creates the SHAP explainer for a given model."""
    return shap.TreeExplainer(_model)


model_median, model_peak, scaler_ar, scaler_env, full_data, EXPECTED_COLUMNS = (
    load_artifacts()
)

# --- App Layout ---
st.title("💡 Previsor de Consumo de Energia")

if model_median is None:
    st.stop()

st.markdown(
    """
Bem-vindo à ferramenta interativa de previsão de consumo de energia. Este aplicativo utiliza um sofisticado pipeline de aprendizado de máquina para prever o uso de energia de eletrodomésticos. Em vez de uma única previsão, muitas vezes imprecisa, usamos uma estratégia de **Regressão Quantílica** para fornecer duas previsões distintas:

- **Previsão Mediana (50º Percentil):** Uma estimativa confiável do consumo *típico* de energia.
- **Previsão de Pico (90º Percentil):** Uma previsão importante que indica possíveis picos de consumo, ajudando a antecipar gastos elevados.

Use a barra lateral para selecionar um horário do período de teste e veja como os modelos se comportam em relação aos dados reais.
"""
)

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Opções de Previsão")

    test_period_start_index = int(len(full_data) * 0.85)
    test_period_start = full_data["date"].iloc[test_period_start_index]
    test_period_end = full_data["date"].iloc[-1]

    selected_date = st.slider(
        "Selecione uma Data e Hora",
        min_value=test_period_start.to_pydatetime(),
        max_value=test_period_end.to_pydatetime(),
        value=test_period_start.to_pydatetime(),
        step=pd.Timedelta(minutes=10),
        format="YYYY-MM-DD HH:mm",
        help="Escolha um momento para gerar a previsão. O app usará as 24 horas anteriores para fazer a previsão.",
    )

# --- Prediction Logic ---
try:
    input_row_index = full_data[full_data["date"] == selected_date].index[0]
    if input_row_index < 144:
        st.warning(
            "Selected date is too early to generate all historical features. Please select a later date."
        )
        st.stop()
    history_df = full_data.iloc[input_row_index - 144 : input_row_index + 1].copy()
except IndexError:
    st.error("Could not find the selected date in the dataset. Please refresh.")
    st.stop()

fe = FeatureEngineer(history_df.set_index("date"))
df_featured = (
    fe.create_time_features()
    .create_cyclical_features()
    .create_lag_and_rolling_features()
    .df
)
if df_featured.empty:
    st.error("Could not generate features for the selected timestamp.")
    st.stop()
prediction_input = df_featured.iloc[[-1]]

if "lights" in prediction_input.columns:
    prediction_input = prediction_input.drop(columns=["lights"])
X_pred = prediction_input[EXPECTED_COLUMNS]

# Scale features in groups
autoregressive_features = [col for col in EXPECTED_COLUMNS if "Appliances_" in col]
environmental_features = [
    col for col in EXPECTED_COLUMNS if col not in autoregressive_features
]
X_pred_ar_scaled = pd.DataFrame(
    scaler_ar.transform(X_pred[autoregressive_features]),
    columns=autoregressive_features,
    index=X_pred.index,
)
X_pred_env_scaled = pd.DataFrame(
    scaler_env.transform(X_pred[environmental_features]),
    columns=environmental_features,
    index=X_pred.index,
)
X_pred_scaled = pd.concat([X_pred_ar_scaled, X_pred_env_scaled], axis=1)[
    EXPECTED_COLUMNS
]

# Make predictions and inverse transform
pred_log_median = model_median.predict(X_pred_scaled)[0]
pred_log_peak = model_peak.predict(X_pred_scaled)[0]
pred_final_median = max(0, np.expm1(pred_log_median))
pred_final_peak = max(0, np.expm1(pred_log_peak))
actual_value = history_df.iloc[-1]["Appliances"]

# --- Display Results ---
st.header(f"Previsão para: {selected_date.strftime('%Y-%m-%d %H:%M')}")
col1, col2, col3 = st.columns(3)
col1.metric("Valor Real (Wh)", f"{actual_value:.2f}")
col2.metric(
    "Previsão Mediana (50º %)",
    f"{pred_final_median:.2f}",
    delta=f"{round(pred_final_median - actual_value, 2)}",
)
col3.metric(
    "Previsão de Pico (90º %)",
    f"{pred_final_peak:.2f}",
    delta=f"{round(pred_final_peak - actual_value, 2)}",
)

with st.expander("O que significam essas previsões?"):
    st.markdown(
        """
    - A **Previsão Mediana** é a melhor estimativa do modelo para o consumo *mais provável* de energia. Você pode ver que ela acompanha bem o consumo base.
    - A **Previsão de Pico** é uma estimativa para valores altos. Ela é sensível a condições que levam a picos de consumo. Um valor alto aqui serve como um importante sinal de alerta.
    """
    )

st.subheader("Previsão em Contexto")
history_to_plot = history_df.set_index("date").iloc[-144:]
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    history_to_plot.index,
    history_to_plot["Appliances"],
    label="Histórico Real",
    color="black",
    alpha=0.7,
)
ax.axvline(selected_date, color="gray", linestyle="--", label="Hora da Previsão")
ax.plot(
    selected_date,
    pred_final_median,
    "o",
    color="dodgerblue",
    markersize=8,
    label="Previsão Mediana",
)
ax.plot(
    selected_date,
    pred_final_peak,
    "o",
    color="orangered",
    markersize=8,
    label="Previsão de Pico",
)
ax.plot(
    selected_date,
    actual_value,
    "o",
    color="limegreen",
    markersize=8,
    label="Valor Real",
)
ax.set_title("Previsões de Consumo de Energia vs. Valores Reais")
ax.set_ylabel("Energia dos Eletrodomésticos (Wh)")
ax.legend()
plt.tight_layout()
st.pyplot(fig)


# --- SHAP Plot Display ---
def shap_plot(explainer, shap_values, features):
    """Helper function to create and robustly display a SHAP force plot."""
    # Let shap create the plot on the global figure context
    shap.force_plot(
        explainer.expected_value,
        shap_values,
        features,
        matplotlib=True,
        show=False,
        text_rotation=10,
    )
    # Grab the current figure that SHAP just created and pass it to Streamlit
    st.pyplot(plt.gcf(), bbox_inches="tight", clear_figure=True)

st.subheader("Por que os modelos fizeram essas previsões?")
st.markdown(
    "Estes **gráficos SHAP** mostram o impacto de cada variável na previsão final. Variáveis em **vermelho** aumentaram a previsão, enquanto variáveis em **azul** diminuíram. Isso mostra como o modelo 'pensa'."
)
col_shap1, col_shap2 = st.columns(2)

with col_shap1:
    st.markdown("##### Explicação da Previsão Mediana")
    explainer_median = get_shap_explainer(model_median)
    shap_values_median = explainer_median.shap_values(X_pred_scaled)
    shap_plot(explainer_median, shap_values_median[0, :], X_pred)

with col_shap2:
    st.markdown("##### Explicação da Previsão de Pico")
    explainer_peak = get_shap_explainer(model_peak)
    shap_values_peak = explainer_peak.shap_values(X_pred_scaled)
    shap_plot(explainer_peak, shap_values_peak[0, :], X_pred)
