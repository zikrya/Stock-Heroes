import matplotlib.pyplot as plt
import numpy as np

def mean_absolute_error(actuals, predictions):
    return np.mean(np.abs(predictions - actuals))

def r2_score(actuals, predictions):
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate_model(predictions, actuals):
    print("Evaluating Model...")
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(actuals, label='Actual Values')
    plt.plot(predictions, label='Predictions')
    plt.title('Model Predictions vs Actual Data')
    plt.legend()
    plt.show()
