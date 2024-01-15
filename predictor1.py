import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print('Mean Squared:', mse)
    print('Root Mean Squared Error:', rmse)
    print('R2 Score:', r2)

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true.index, y_true, color='blue', label='Actual', alpha=0.5)
    plt.scatter(y_true.index, y_pred, color='red', label='Predicted', alpha=0.5)
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Gold Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

start_date = '2011-01-01'
end_date = pd.Timestamp.today().today().strftime('%Y-%m-%d')
print(end_date)
gold_data = yf.download('GC=F', start=start_date, end=end_date)
y = gold_data['Close']
X = np.arange(len(y)).reshape(-1, 1)  # Using the index as a feature for simplicity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X)
calculate_metrics(y, pred_lr)
plot_actual_vs_predicted(y, pred_lr, 'Linear Regression')

# Lasso
la = Lasso()
la.fit(X_train, y_train)
pred_la = la.predict(X)
calculate_metrics(y, pred_la)
plot_actual_vs_predicted(y, pred_la, 'Lasso')

# Ridge
ri = Ridge()
ri.fit(X_train, y_train)
pred_ri = ri.predict(X)
calculate_metrics(y, pred_ri)
plot_actual_vs_predicted(y, pred_ri, 'Ridge')

svr = SVR()
params = {'C': [1, 10, 100, 1000],'kernel': ['rbf'], 'gamma': [1, 10, 100, 1000]}
grid = GridSearchCV(SVR(), params, refit=True, verbose=3)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
# SVR
svr = SVR(kernel='rbf', C=100, gamma=0.0001)
svr.fit(X_train, y_train)
pred_svr = svr.predict(X)
calculate_metrics(y, pred_svr)
plot_actual_vs_predicted(y, pred_svr, 'SVR')

# Predict the next ten days
future_dates = 10
future_date_range = pd.date_range(start=gold_data.index[-1], periods=future_dates + 1, freq='B')  # Change the frequency to 'B'
X_future = np.arange(len(gold_data), len(gold_data) + future_dates).reshape(-1, 1)
pred_svr_future = svr.predict(X_future)

# Combine historical and future dates
combined_date_range = gold_data.index.union(future_date_range)

# Ensure lengths match
assert len(y) == len(pred_svr)
assert len(np.nan * np.ones(len(pred_svr_future))) == len(pred_svr_future)

# Create DataFrame
predictions_df = pd.DataFrame({
    'Date': combined_date_range,
    'Actual Price': np.concatenate([y, np.nan * np.ones(len(pred_svr_future))]),
    'Predicted Price': np.concatenate([pred_svr, pred_svr_future]),
})

# Calculate the difference for historical data
predictions_df['Difference'] = predictions_df['Actual Price'] - predictions_df['Predicted Price']

# Display the DataFrame
print(predictions_df.tail(20))
