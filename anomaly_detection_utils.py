import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    cat = df.select_dtypes(exclude=["number"])
    cat.pop("Timestamp")

    num = df.select_dtypes(include=["number"], exclude=['int64'])

    frame_no = df.pop('Frame No')

    timestamp = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df.pop('Timestamp')

    def label_encode_categorical(df):
        label_encoder = LabelEncoder()
        encoded_df = df.apply(label_encoder.fit_transform)
        return encoded_df

    encoded_cat = label_encode_categorical(cat)

    alpha = 2
    beta = 3

    def calculate_bounds(data, alpha, beta):
        # Calculate bounds for outlier removal
        P_upper = np.percentile(data, 100-alpha)
        P_lower = np.percentile(data, alpha)
        theta_upper = P_upper + beta * (P_upper - P_lower)
        theta_lower = P_lower - beta * (P_upper - P_lower)
        return theta_lower, theta_upper

    def replace_outliers_with_previous(series, alpha, beta):
        # Calculate bounds for the series
        theta_lower, theta_upper = calculate_bounds(series, alpha, beta)
        
        # Replace outliers with the previous value
        series[(series < theta_lower) | (series > theta_upper)] = np.nan
        series.fillna(method='ffill', inplace=True)
        
        return series

    def replace_outliers_in_dataframe(df, alpha, beta):
        # Apply replace_outliers_with_previous to each column
        replaced_df = df.apply(lambda column: replace_outliers_with_previous(column, alpha, beta))
        
        return replaced_df

    # Replace outliers with the previous value in the combined dataframe
    replaced_df_num = replace_outliers_in_dataframe(num, alpha, beta)
    replaced_df_cat = replace_outliers_in_dataframe(encoded_cat, alpha, beta)

    def normalize_data(df, alpha, num_cols_to_normalize=12):
        df_normalized = pd.DataFrame()
        for i, column in enumerate(df.columns):
            if np.issubdtype(df[column].dtype, np.number) and i < num_cols_to_normalize:
                p_high = np.percentile(df[column], 100 - alpha)
                p_low = np.percentile(df[column], alpha)
                y_avg = np.mean(df[column])
                df_normalized[column] = 2 * (df[column] - y_avg) / (p_high - p_low)
            else:
                # If the column is either categorical or beyond the first 12 numerical columns, keep it as is
                df_normalized[column] = df[column]
        return df_normalized

    # Normalize the first 12 numerical columns in the cleaned combined data
    normalized_combined_data = normalize_data(replaced_df_num, 1)

    combined_df = pd.concat([num, encoded_cat], axis=1)

    num_train = normalized_combined_data[:109367]
    cat_train = encoded_cat[:109367]
    frame_no_train = frame_no[:109367]
    timestamp_train = timestamp[:109367]

    num_test = normalized_combined_data[109367:]
    cat_test = encoded_cat[109367:]
    frame_no_test = frame_no[109367:]
    timestamp_test = timestamp[109367:]

    return num_train, cat_train, frame_no_train, timestamp_train, num_test, cat_test, frame_no_test, timestamp_test,combined_df
