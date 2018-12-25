import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse


"""
    Demo for showing bias-variance decomposition. During the following proof 
    (http://www.inf.ed.ac.uk/teaching/courses/mlsc/Notes/Lecture4/BiasVariance.pdf)
    some terms related to epsilon (noise) are set to 0 because E(epsilon) = 0. Of course in real data E(epsilon) it is 
    not exactly 0.
    In this demo we show how the expected squared error is equal to bias_squared + variancce + noise_variance + 
    remaining_errors_due_to_noise
    When n_trials (numbers of measurements on fx) is very large then the last term can be neglected. For this demo we 
    consider this term for academic purposes.
    
    We show bias-variance decomposition on the sin(x) function and observe how the error components evolve as the 
    complexity of the model changes.
    
"""


np.random.seed(2)
n_trials = 50
noise_std = 0.5
max_poly = 12
n_total = 25
n_train = int(0.9 * n_total)
train_ind = np.random.choice(n_total, n_train, replace = False);
test_ind = np.array([i for i in range(n_total) if i not in train_ind]);

# Feature engineering. Add D powers to datapoint x so to create models with different complexities.
def make_poly(x, D):
    m = len(x)
    X = np.empty((m, D + 1))
    for d in range(D + 1):
        X[:, d] = x ** d
        if d > 1:
            # Normalize the curves for them not to take large values
            X[:, d] = (X[:, d] - X[:, d].mean()) / X[:, d].std()
    return X


def sin_dataset():
    X = np.linspace(-np.pi, np.pi, n_total)
    fx = np.sin(X)
    # Each row is x_i, x_i**2, x_i**3, x_i**4, ..., x_i**12
    X_poly = make_poly(X, max_poly)
    return X_poly, fx;

def make_target_sets(fx):
    target_sets = np.zeros((n_total, n_trials))
    noise = np.zeros((n_total, n_trials))
    for k in range(n_trials):
        noise[:, k] = np.random.randn(n_total) * noise_std;
        target_sets[:, k] = fx + noise[:, k]
    return noise, target_sets;

def train_models(X_poly, target_sets):
    train_scores = np.zeros((n_trials, max_poly))
    test_scores = np.zeros((n_trials, max_poly))

    y_train_per_dataset = np.zeros((n_train, n_trials))

    y_train_predictions = np.zeros((n_train, n_trials, max_poly))

    # create the model
    model = LinearRegression()

    # A dataset is simply fx + a set noise
    # So at this point X is the same, what changes is the target
    for k in range(n_trials):
        y = target_sets[:, k]
        X_train = X_poly[train_ind]
        # (22, )
        y_train = y[train_ind]
        y_train_per_dataset[:, k] = y_train;

        X_test = X_poly[test_ind]
        y_test = y[test_ind]

        # Fit model with complexity d on dataset k
        for d in range(max_poly):
            model.fit(X_train[:, :d + 2], y_train_per_dataset[:, k])

            # Get predictions of model d for dataset k
            y_train_pred = model.predict(X_train[:, :d + 2])
            y_test_pred = model.predict(X_test[:, :d + 2])

            # Use this to calculate bias/variance later
            y_train_predictions[:, k, d] = y_train_pred

            train_score = mse(y_train_pred, y_train)
            test_score = mse(y_test_pred, y_test)

            # Save train and test scores for model d and dataset k
            train_scores[k, d] = train_score
            test_scores[k, d] = test_score

    return y_train_per_dataset, y_train_predictions, train_scores, test_scores


def make_bias_variance_decomposition(fx, noise, y_train_predictions):
    # Calculate the expected (y_i - p_i)**2 using the bias-variance decomposition terms.

    # n_train total training points
    avg_train_prediction = np.zeros((n_train, max_poly))
    squared_bias = np.zeros((n_train, max_poly))
    variance = np.zeros((n_train, max_poly))
    noise_variance = np.zeros((n_train, max_poly))
    err1 = np.zeros((n_train, max_poly))
    err2 = np.zeros((n_train, max_poly))

    fx_train = fx[train_ind];
    noise_train = noise[train_ind, :];

    # Calculate the bias
    for d in range(max_poly):  # For model with complexity d
        for i in range(n_train):  # Look at f_i
            # Mean over the predictions made by model with complexity d on point i.
            # Remember this is not a single model but k models (that were trained on k different datasets) with complexity d
            avg_train_prediction[i, d] = y_train_predictions[i, :, d].mean()
            squared_bias[i, d] = (avg_train_prediction[i, d] - fx_train[i]) ** 2

    # Calculate the variance
    for d in range(max_poly):
        for i in range(n_train):
            delta = y_train_predictions[i, :, d] - avg_train_prediction[i, d]
            variance[i, d] = np.mean(delta ** 2)

    # Calculate the variance of the noise
    # The noise is independent of the model and the predictions
    for i in range(n_train):
        noise_variance[i, :] = np.mean(noise_train[i, :] ** 2)

    # Let's calculate terms containing epsilon(noise error) derivated when expanding E((y_i - p_i)**2)
    # Their expectation is 0. However, we calculate them to check the correcness of the bias-variance formulation.
    for d in range(max_poly):
        for i in range(n_train):
            prod = noise_train[i] * y_train_predictions[i, :, d]
            err1[i, d] = -2 * np.mean(prod)

    for i in range(n_train):
        err2[i, :] = 2 * fx_train[i] * np.mean(noise_train[i, :])


    return {"bias_squared": squared_bias,
            "variance": variance,
            "noise_variance": noise_variance,
            "err1": err1,
            "err2": err2}

def calculate_error_without_decomposition(y_train_per_dataset, y_train_predictions):
    # Calculate independently the expected (y_i - p_i)**2
    expected_error = np.zeros((n_train, max_poly))
    for d in range(max_poly):
        for i in range(n_train):
            delta = (y_train_per_dataset[i, :] - y_train_predictions[i, :, d]) ** 2
            expected_error[i, d] = np.mean(delta)

    return expected_error;


def plot_error_on_point(decomposition_terms, expected_error, i):
    degrees = np.arange(max_poly) + 1

    bias_squared = decomposition_terms["bias_squared"]
    variance = decomposition_terms["variance"]
    noise_variance = decomposition_terms["noise_variance"]
    err1 = decomposition_terms["err1"]
    err2 = decomposition_terms["err2"]

    e_err = expected_error[i, :]

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(right=0.6)
    plt.title("Bias-Variance decomposition for E((y_{} - p_{})**2) (for datapoint x_{})".format(i, i, i))

    plt.plot(degrees, bias_squared[i, :], label='squared bias')
    plt.plot(degrees, variance[i, :], label='variance')
    plt.plot(degrees, noise_variance[i, :], label='noise squared')
    plt.plot(degrees, err1[i, :] + err2[i, :], label='remaining error')

    sum_error = bias_squared[i, :] + variance[i, :] + noise_variance[i, :] + err1[i, :] + err2[i, :];

    plt.plot(degrees, sum_error, 'bo', label='Expected error with b-v decomposition')
    plt.plot(degrees, e_err, label="Expected error without decomposition", linestyle="--")
    plt.legend(bbox_to_anchor=(1, 0.7))
    plt.xlabel("Model complexity")
    plt.ylabel("Expected SE")
    plt.show()

def plot_mean_error(decomposition_terms, expected_error):
    # Do the same but with all datapoints. To do so we just take the mean of the corresponding errors over all datapoints.

    # make bias-variance plots
    degrees = np.arange(max_poly) + 1

    sqb = decomposition_terms["bias_squared"].mean(axis=0);
    var = decomposition_terms["variance"].mean(axis=0);
    nsq = decomposition_terms["noise_variance"].mean(axis=0)
    err1 = decomposition_terms["err1"].mean(axis = 0)
    err2 = decomposition_terms["err2"].mean(axis = 0)

    e_err = expected_error.mean(axis=0);

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(right=0.6)
    plt.title("Bias-Variance decomposition for E(MSE) (all datapoints)")
    plt.plot(degrees, sqb, label='squared bias')
    plt.plot(degrees, var, label='variance')
    plt.plot(degrees, nsq, label='noise squared')
    plt.plot(degrees, err1 + err2, label='rem error')

    errorsum = sqb + var + nsq + err1 + err2;

    plt.plot(degrees, errorsum, 'bo', label='Expected error with b-v decomposition')
    plt.plot(degrees, e_err, label="Expected error without decomposition", linestyle="--")
    plt.legend(bbox_to_anchor=(1, 0.7))
    plt.xlabel("Model complexity")
    plt.ylabel("E(MSE)")
    plt.show()


def plot_train_test_error(train_scores, test_scores):
    degrees = np.arange(max_poly) + 1
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(right=0.6)
    plt.title("Train-test E(MSE) error")
    best_degree = np.argmin(test_scores.mean(axis=0)) + 1
    plt.plot(degrees, train_scores.mean(axis=0), label='Train scores')
    plt.plot(degrees, test_scores.mean(axis=0), label='Test scores')
    plt.axvline(x=best_degree, linestyle='--', label='Best complexity')
    plt.legend(bbox_to_anchor=(1, 0.7))
    plt.xlabel("Model complexity")
    plt.ylabel("E(MSE)")
    plt.show()


def main():

    #Create sin(x) dataset
    X_poly, fx = sin_dataset()

    #Create "n_trials" sets of targets on fx
    noise, target_sets = make_target_sets(fx)

    #Train models over all datasets k and model complexities d
    y_train_per_dataset, y_train_predictions, train_scores, test_scores = train_models(X_poly, target_sets)

    #Calculate the expected squared error for each point in the training set using decomposition
    error_decomposition = make_bias_variance_decomposition(fx, noise, y_train_predictions)

    #Calculate the expected squared error for each point in the training set without decomposition
    expected_error = calculate_error_without_decomposition(y_train_per_dataset, y_train_predictions)

    #Plot the error decomposition for datapoint 0 (Can be any point from 0 to n_train)
    plot_error_on_point(error_decomposition, expected_error, 0);

    # Plot the error decomposition over all datapoints E(MSE)
    plot_mean_error(error_decomposition, expected_error);

    #Finally plot the error curves for the training dataset and test dataset
    plot_train_test_error(train_scores, test_scores);

if __name__ == "__main__":
    main();