import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess dataset
@st.cache_data  # ✅ Updated to prevent Streamlit deprecation warning
def load_data():
    df = pd.read_csv("C:/Users/jyots/OneDrive/Desktop/Women-Health/CLEAN- PCOS SURVEY SPREADSHEET.csv")  # Ensure correct file path
    df = df.dropna()  # Remove missing values

    # ✅ Correct column names (underscore instead of spaces)
    selected_features = ["Age", "Weight_Gain", "Hair_Loss", "Acne", "Regular_Periods", "Exercise"]

    # ✅ Check if all required columns exist
    missing_columns = [col for col in selected_features if col not in df.columns]
    if missing_columns:
        st.error(f"Error: Missing columns in dataset: {missing_columns}")
        st.stop()  # Stop execution if essential columns are missing

    df = df[selected_features]
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=selected_features)
    return df, df_scaled, scaler

df, df_scaled, scaler = load_data()


# Train clustering models
hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')  # ✅ Fixed `affinity` error
hc_clusters = hc.fit(df_scaled).labels_

gmm = GaussianMixture(n_components=4, random_state=42)
gmm_clusters = gmm.fit_predict(df_scaled)

df["HC_Cluster"] = hc_clusters
df["GMM_Cluster"] = gmm_clusters


def predict_cluster(user_data):
    """Predict the cluster for user input based on trained models."""
    X = pd.DataFrame([user_data], columns=["Age", "Weight_Gain", "Hair_Loss", "Acne", "Regular_Periods", "Exercise"])
    X_scaled = scaler.transform(X)

    # GMM can predict directly
    gmm_cluster = gmm.predict(X_scaled)[0]

    # AgglomerativeClustering does not support .predict(), so find closest cluster
    hc_cluster = min(range(4), key=lambda c: np.linalg.norm(X_scaled - df_scaled[df["HC_Cluster"] == c].mean().values))

    return hc_cluster, gmm_cluster

def display_recommendations(cluster_id):
    """Display recommendations based on cluster."""
    recommendations = {
        0: "**Balanced Diet, Moderate Exercise, Annual Check-Ups, Stress Management**\n\n"
           "- **Balanced Diet**: Consume a variety of nutrient-dense foods like whole grains, lean proteins, and healthy fats while avoiding processed foods and excess sugar.\n"
           "- **Moderate Exercise**: Engage in at least 150 minutes of moderate aerobic activity weekly, such as walking or cycling.\n"
           "- **Annual Check-Ups**: Schedule regular screenings to detect any health issues early.\n"
           "- **Stress Management**: Practice mindfulness, meditation, or yoga to reduce stress and improve mental well-being.",
        
        1: "**Low Glycemic Index (GI) Diet, Strength Training, Hormonal Tests, Inositol Supplements**\n\n"
           "- **Low GI Diet**: Focus on whole grains, legumes, and fiber-rich foods to stabilize blood sugar levels.\n"
           "- **Strength Training**: Engage in resistance training exercises to improve insulin sensitivity and metabolic health.\n"
           "- **Hormonal Tests**: Regularly monitor insulin, testosterone, and thyroid hormone levels to manage conditions like PCOS.\n"
           "- **Inositol Supplements**: Myo-inositol and D-chiro-inositol can help regulate hormones and improve insulin resistance.",
        
        2: "**Anti-Inflammatory Diet, Regular Glucose Monitoring, Berberine Supplement**\n\n"
           "- **Anti-Inflammatory Diet**: Consume omega-3-rich foods (salmon, walnuts), leafy greens, and berries while avoiding trans fats and processed foods.\n"
           "- **Regular Glucose Monitoring**: Keep track of blood sugar levels to prevent insulin resistance.\n"
           "- **Berberine Supplement**: This natural compound may help regulate glucose metabolism and improve insulin sensitivity.",
        
        3: "**Maintain Healthy Habits, Routine Health Screenings, Mindfulness Practices**\n\n"
           "- **Healthy Habits**: Continue with a nutritious diet and regular exercise to sustain long-term well-being.\n"
           "- **Routine Health Screenings**: Stay proactive about health check-ups to detect early signs of potential issues.\n"
           "- **Mindfulness Practices**: Engage in meditation, yoga, or deep breathing exercises to manage stress effectively."
    }
    st.write(f"### Personalized Recommendation for Your Cluster ({cluster_id}):")
    st.info(recommendations.get(cluster_id, "No recommendation available."))

# Streamlit UI
st.title("Women's Health Clustering & Personalized Recommendations")
st.write("Enter your health details to receive a personalized recommendation.")

age = st.number_input("Age", min_value=10, max_value=60, value=25)
weight_gain = st.slider("Weight Gain (1-10)", 1, 10, 5)
hair_loss = st.slider("Hair Loss (1-10)", 1, 10, 5)
acne = st.slider("Acne Severity (1-10)", 1, 10, 5)
period_regularity = st.slider("Period Regularity (1-10)", 1, 10, 5)
exercise = st.slider("Exercise Frequency (Days per Week)", 0, 7, 3)

if st.button("Get My Cluster & Recommendations"):
    user_data = [age, weight_gain, hair_loss, acne, period_regularity, exercise]
    hc_cluster, gmm_cluster = predict_cluster(user_data)
    st.write(f"You belong to GMM Cluster: **{gmm_cluster}**")
    display_recommendations(gmm_cluster)
