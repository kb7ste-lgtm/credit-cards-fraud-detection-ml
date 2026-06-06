import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

# 1) Configuration de page
st.set_page_config(page_title="Fraud Detection App", layout="wide")

# 2) Chargement du pipeline
@st.cache_resource
def load_production_pipeline():
    with open("cartes_credits_fraud_detection_lgbm.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_amount.pkl", "rb") as f:
        scaler_amount = pickle.load(f)
    with open("scaler_time.pkl", "rb") as f:
        scaler_time = pickle.load(f)
    return model, scaler_amount, scaler_time

pipeline_prod, scaler_amount, scaler_time = load_production_pipeline()

# 3) Interface utilisateur
header_container = st.container()
with header_container:
    st.markdown("""
        <div style="background-color: #003366; padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h1 style="margin: 0;">Credit Card Fraud Detection</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Structure 3 colonnes pour aligner la carte (gauche) et le logo (droite)
    # Le ratio [2, 5, 2] crée un grand espace vide au milieu qui pousse le logo à droite
    col1, col2, col3 = st.columns([2, 5, 2])
    
    with col1:
        st.image("fraud_image.png", width=300)
    with col3:
        st.image("ehtp_logo.png", width=250)

st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Transaction Inputs")
    Time = st.slider("Time", 0.0, 172792.0, 0.0)
    Amount = st.slider("Amount", 0.0, 5000.0, 10.0)
    
    st.markdown("---")
    st.subheader("Composantes PCA (V1 à V28)")
    
    # Boucle pour générer les sliders proprement
    pca_inputs = {}
    for i in range(1, 29):
        pca_inputs[f"V{i}"] = st.slider(f"V{i}", -30.0, 30.0, 0.0)

# 4) RobustScaler
amount_scaled = scaler_amount.transform([[Amount]])[0][0]
time_scaled = scaler_time.transform([[Time]])[0][0]

# 5) Dataframe
input_data = {**pca_inputs, "amount_scaled": amount_scaled, "time_scaled": time_scaled}
input_df = pd.DataFrame([input_data])

st.subheader("User Input Parameters (Sent to Model)")
st.write(input_df)

# 6) Bouton predict
if st.button("Predict"):
    if pipeline_prod is not None:
        prediction = pipeline_prod.predict(input_df)
        proba = pipeline_prod.predict_proba(input_df)

        st.subheader("Result")
        if prediction[0] == 1:
            st.error(f"🚨 ALERT: Fraudulent Transaction Detected! (Confidence: {proba[0][1]*100:.2f}%)")
        else:
            st.success(f"🎉 Transaction Approved (Legitimate) (Confidence: {proba[0][0]*100:.2f}%)")

        st.subheader("Prediction Probabilities")
        st.write(pd.DataFrame(proba, columns=["Legitimate (0)", "Fraudulent (1)"]))

# 7) Predict via csv
st.markdown("---")
st.header("📁 Predict via CSV File")

uploaded_file = st.file_uploader("Upload your transactions CSV file here", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.write("Preview:", df_raw.head())
    
    if st.button("Run CSV Prediction"):
        df_process = df_raw.copy()
        df_process['amount_scaled'] = scaler_amount.transform(df_process['Amount'].values.reshape(-1, 1)).reshape(-1)
        df_process['time_scaled'] = scaler_time.transform(df_process['Time'].values.reshape(-1, 1)).reshape(-1)
        df_process.drop(['Time', 'Amount'], axis=1, inplace=True)
        
        columns_order = [f"V{i}" for i in range(1, 29)] + ["amount_scaled", "time_scaled"]
        df_process = df_process[columns_order]
        
        df_raw['Prediction'] = pipeline_prod.predict(df_process)
        st.write("Results:", df_raw)