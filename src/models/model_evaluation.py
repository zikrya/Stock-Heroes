import matplotlib.pyplot as plt
import numpy as np

def mean_absolute_error(actuals, predictions):
    return np.mean(np.abs(predictions - actuals))

def r2_score(actuals, predictions):
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate_model(predictions, actuals):
    error = predictions - actuals
    mae = np.mean(np.abs(error))
    mse = np.mean(np.square(error))
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(error, bins=50, alpha=0.7, color='red', label='Prediction Error')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(actuals, predictions, alpha=0.7, color='blue')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=4)
    plt.show()