import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import streamlit as st

class AnomalyDetectionModel:
    def __init__(self):
        pass

    def apply_pca(self, X_cont, n_pca_components):
        pca = PCA(n_components=n_pca_components)
        X_cont_pca = pca.fit_transform(X_cont)
        return X_cont_pca, pca
    
    with st.spinner("Intializing the parameters"):
        def initialize_parameters_ppca(self, X_cont_pca, n_clusters, n_components):
            st.write("")
            n_samples, n_pca_features = X_cont_pca.shape

            random_indices_cont = np.random.choice(n_samples, n_clusters, replace=False)
            means_cont = X_cont_pca[random_indices_cont, :]
            covariances_cont = [np.eye(n_pca_features) for _ in range(n_clusters)]
            weights = np.ones(n_clusters) / n_clusters
            W = np.random.randn(n_clusters, n_pca_features, n_components)
            variances_cont = np.ones(n_clusters)

            return means_cont, covariances_cont, weights, W, variances_cont
    st.success('Parameters Initialization completed')

    def initialize_parameters_categorical(self, X_cat_encoded, n_clusters):
        n_samples, n_cat_features = X_cat_encoded.shape
        theta = np.random.dirichlet(np.ones(X_cat_encoded.max() + 1), size=(n_clusters, n_cat_features))
        return theta
    
    with st.spinner("Checking Convergence"):
        def check_convergence(self, means_cont, new_means_cont, covariances_cont, new_covariances_cont,
                            weights, new_weights, W, new_W, variances_cont, new_variances_cont, tol=1e-4):
            
            mean_diff_cont = np.linalg.norm(new_means_cont - means_cont)
            cov_diff_cont = sum(np.linalg.norm(new_cov - old_cov) for new_cov, old_cov in zip(new_covariances_cont, covariances_cont))
            weight_diff = np.linalg.norm(new_weights - weights)
            W_diff_cont = np.linalg.norm(new_W - W)
            var_diff_cont = np.linalg.norm(new_variances_cont - variances_cont)

            return (mean_diff_cont < tol) and (cov_diff_cont < tol) and (weight_diff < tol) and (W_diff_cont < tol) and (var_diff_cont < tol)
    st.success('Convergence Detected')

    def e_step_ppca(self, X_cont, means_cont, covariances_cont, weights, W, variances_cont):
        n_samples, n_clusters, n_features = X_cont.shape[0], weights.shape[0], means_cont.shape[1]

        cont_resp = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            covar = W[k] @ W[k].T + variances_cont[k] * np.eye(n_features)
            rv = multivariate_normal(mean=means_cont[k], cov=covar)
            cont_resp[:, k] = rv.pdf(X_cont) * weights[k]

        resp_sum = cont_resp.sum(axis=1, keepdims=True)
        resp = cont_resp / resp_sum

        return resp

    def e_step_categorical(self, X_cat_encoded, theta):
        n_samples, n_clusters, n_cat_features = X_cat_encoded.shape[0], theta.shape[0], theta.shape[1]

        cat_resp = np.ones((n_samples, n_clusters))
        for feature in range(n_cat_features):
            feature_values = X_cat_encoded[:, feature]
            for cluster in range(n_clusters):
                cat_resp[:, cluster] *= theta[cluster, feature, feature_values]

        resp_sum = cat_resp.sum(axis=1, keepdims=True)
        resp = cat_resp / resp_sum

        return resp

    def m_step_ppca(self, X_cont, resp, means_cont):
        n_samples, n_clusters, n_features = X_cont.shape[0], resp.shape[1], means_cont.shape[1]

        new_means_cont = np.zeros((n_clusters, n_features))
        new_covariances_cont = [np.zeros((n_features, n_features)) for _ in range(n_clusters)]
        new_variances_cont = np.zeros(n_clusters)
        new_weights = resp.sum(axis=0) / n_samples

        for k in range(n_clusters):
            Nk = resp[:, k].sum()
            new_means_cont[k] = np.dot(resp[:, k], X_cont) / Nk

            X_centered = X_cont - new_means_cont[k]
            covar_k = np.dot(X_centered.T, X_centered * resp[:, k][:, np.newaxis]) / Nk
            new_covariances_cont[k] = covar_k
            new_variances_cont[k] = np.trace(covar_k) / n_features

        return new_means_cont, new_covariances_cont, new_variances_cont, new_weights

    def m_step_categorical(self, X_cat_encoded, resp):
        n_samples, n_clusters, n_cat_features = X_cat_encoded.shape[0], resp.shape[1], X_cat_encoded.shape[1]

        new_theta = np.zeros((n_clusters, n_cat_features, X_cat_encoded.max() + 1))

        for k in range(n_clusters):
            for feature in range(n_cat_features):
                cat_counts = np.zeros(X_cat_encoded.max() + 1)
                for cat_value in range(X_cat_encoded.max() + 1):
                    cat_counts[cat_value] = resp[:, k][X_cat_encoded[:, feature] == cat_value].sum()

                new_theta[k, feature] = (cat_counts + 1) / (np.sum(resp[:, k]) + X_cat_encoded.max() + 1)

        return new_theta

    with st.spinner("Running the EmAlgorithm"):
        def em_algorithm(self, X_cont, X_cat_encoded, n_clusters, n_components, n_pca_components, max_iter=100, tol=1e-4):
            X_cont_pca, pca_model = self.apply_pca(X_cont, n_pca_components)
            means_cont, covariances_cont, weights, W, variances_cont = self.initialize_parameters_ppca(X_cont_pca, n_clusters, n_components)
            theta = self.initialize_parameters_categorical(X_cat_encoded, n_clusters)
            for iteration in range(max_iter):
                resp_ppca = self.e_step_ppca(X_cont_pca, means_cont, covariances_cont, weights, W, variances_cont)
                resp_cat = self.e_step_categorical(X_cat_encoded, theta)
                responsibilities = resp_ppca * resp_cat

                new_means_cont, new_covariances_cont, new_variances_cont, new_weights = self.m_step_ppca(X_cont_pca, responsibilities, means_cont)
                new_theta = self.m_step_categorical(X_cat_encoded, responsibilities)

                if self.check_convergence(means_cont, new_means_cont, covariances_cont, new_covariances_cont,weights, new_weights, W, variances_cont, new_variances_cont, 1e-4):
                    break

                means_cont, covariances_cont, weights, W, variances_cont, theta = new_means_cont, new_covariances_cont, new_weights, W, new_variances_cont, new_theta

            return means_cont, covariances_cont, weights, W, variances_cont, theta
    st.success('Success in running the EmAlgorithm')

    with st.spinner("Calculating the Joint Probability"):
        def calculate_joint_probability(self, X_cont, X_cat, pi_k, means_k, covariances_k, psi_j_k):
            n_samples = X_cont.shape[0]
            n_clusters = len(pi_k)
            joint_probabilities = np.zeros(n_samples)
            
            for k in range(n_clusters):
                covariances_k[k] += np.eye(covariances_k[k].shape[0]) * 1e-6
                continuous_density = multivariate_normal.pdf(X_cont, mean=means_k[k], cov=covariances_k[k])
                categorical_probability = np.prod([psi_j_k[k, j, X_cat[:, j]] for j in range(X_cat.shape[1])], axis=0)
                joint_probabilities += pi_k[k] * continuous_density * categorical_probability
            
            return joint_probabilities
    st.success('Joint Probability Calculated')

    with st.spinner("Computing the Log Likelihoods"):
        def compute_log_likelihood(self, joint_probs):
            st.write("Computing the Log Likelihoods")
            log_likelihood = -np.log(joint_probs)
            return log_likelihood
    st.success('success in Calculating the Log Likelihoods')
