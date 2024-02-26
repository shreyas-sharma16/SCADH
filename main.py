import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import plotly.express as px
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from anomaly_detection_utils import preprocess_data
from mppcacd_model import AnomalyDetectionModel

# Model training functions

# Main function
def main():
    st.title("Anomaly Detection")

    st.write("Please select a CSV file from your desktop.")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        st.write("File selected:")
        st.write(uploaded_file.name)

        # Display contents of the CSV file
        df = pd.read_csv(uploaded_file)
        st.write(df)
        with st.spinner('Preprocessing data...'):
            num_train, cat_train, frame_no_train, timestamp_train, num_test, cat_test, frame_no_test, timestamp_test,combined_df = preprocess_data(df)
        st.success('Preprocessing done!')

        mppcacd_selected = st.checkbox("MPPCACA Algorithm")
        lstm_selected = st.checkbox("LSTM Model")

        
        if mppcacd_selected:
            model = AnomalyDetectionModel()
            with st.spinner('Running the MPPCACA model...'):
                X_std = StandardScaler().fit_transform(combined_df)
                pca = PCA().fit(X_std)
                L = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
                pca_L = PCA(n_components=L)
                X_pca_L_train = pca_L.fit_transform(num_train)
                X_pca_L_test = pca_L.fit_transform(num_test)

                n_clusters = 3
                n_components = 3

                final_means, final_covariances, final_weights, final_W, final_variances, final_theta = model.em_algorithm(
                    X_pca_L_train, cat_train.astype("int").to_numpy(), n_clusters, n_components, L)

                joint_probs = model.calculate_joint_probability(X_pca_L_test, cat_test.astype("int").to_numpy(),
                                                                final_weights, final_means, final_covariances, final_theta)

                log_likelihoods = model.compute_log_likelihood(joint_probs)

                anomaly_score_time = pd.DataFrame(timestamp_test).reset_index().join(pd.DataFrame(log_likelihoods, columns=['Log Likelihood']))

            st.success('Hurray!! Anomaly has been detected')
            fig = px.line(anomaly_score_time, x='Timestamp', y='Log Likelihood', title='Log Likelihood vs Timestamp', render_mode='webgl')
            fig.update_xaxes(title_text='Timestamp')
            fig.update_yaxes(title_text='Log Likelihood')
            st.plotly_chart(fig)

            # Convert string timestamp to pandas Timestamp object
            max_index = np.argmax(log_likelihoods)

            # Retrieve the corresponding timestamp (day) only if there are non-zero anomaly scores
            if log_likelihoods[max_index] != 0:
                highest_anomaly_timestamp = timestamp_test.iloc[max_index]
                highest_anomaly_day = pd.to_datetime(highest_anomaly_timestamp).date()
                
                # Print the highest anomaly score and corresponding timestamp (day)
                st.write("Highest anomaly score:", log_likelihoods[max_index])
                st.write("Timestamp (Day) of highest anomaly score:", highest_anomaly_day)
            else:
                st.write("No anomaly scores found.")




if __name__ == "__main__":
    main()