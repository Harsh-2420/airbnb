import pandas as pd
import numpy as np

tr_filepath = "/Users/harshjhunjhunwala/Desktop/github_datasets/airbnb_data/train_users_2.csv"
df_train = pd.read_csv(tr_filepath, header=0, index_col=None)
te_filepath = "/Users/harshjhunjhunwala/Desktop/github_datasets/airbnb_data/test_users.csv"
df_test = pd.read_csv(te_filepath, header=0, index_col=None)
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# Fix the datetime formats in the date column
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d')
df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')
df_all['date_account_created'].fillna(df_all.timestamp_first_active, inplace=True)

# Drop date_first_booking to avoid creating an incorrect model
df_all.drop('date_first_booking', axis=1, inplace=True)

# Fixing age column
def remove_outliers(df, column, min_val, max_val):
    col_values = df[column.values]
    df[column] = np.where(np.logical_or(col_values <= min_val, col_values >= max_val), np.NaN, col_values)
    return df
df_all = remove_outliers(df_all, 'age', 15, 90)
df_all['age'].fillna(-1, inplace=True)
df_all['first_affiliate_tracked'].fillna(-1, inplace=True)