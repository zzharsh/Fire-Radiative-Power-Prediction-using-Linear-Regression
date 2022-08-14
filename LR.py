import pandas as pd
import numpy as np
import matplotlib

#matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        self.mean_set_features = 0
        self.sd_set_features = 0

    def __call__(self, features, is_train=False):
        # Separating the non categorical features
        non_categorical = list((features.drop(['satellite_Terra', 'daynight_N'], axis=1)).columns.values.tolist())

        # If train set, then calculate the mean and sd and then normalize, else normalize using the earlier calculated mean and sd.
        if is_train == True:

            self.mean_set_features = features[non_categorical].mean().astype(float)  # Mean calculated
            self.sd_set_features = features[non_categorical].std().astype(float)  # Standard deviation calculated

            for i in non_categorical:
                features[i] = features[i].apply(lambda x: (x - self.mean_set_features[i]) / self.sd_set_features[i])

        elif is_train == False:

            for i in non_categorical:
                features[i] = features[i].apply(lambda x: (x - self.mean_set_features[i]) / self.sd_set_features[i])

        return features


def get_features(csv_path, is_train=False, scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n
    m is number of examples, n is number of features
    return value: numpy array
    '''

    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''

    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''

    # If training set, then data points for whom frp is less than 3000 will only be considered so as to remove the effect of outliers
    # similarly for the dev/test set, only values below 9000 would be considered. Dev set contained few points at around 10,000 giving unnecessary error.
    data_df = pd.read_csv(csv_path, index_col=0)
    # if is_train == True:
    #     data_df = data_df[data_df["frp"] < 3000]
    # else:
    #     if ('frp' in data_df.columns):
    #         data_df = data_df[data_df["frp"] < 9000]
    #     else:
    #         data_df['frp'] = data_df['version']

    # Defining input as X, dropping below features
    X = data_df.drop(['frp', 'version', 'instrument', 'acq_date', 'acq_time'], axis=1)

    # satellite and daynight into categorical values
    categorical_columns = ['satellite', 'daynight']

    # Dropping the dummy variables to avoid dummy variable trap,
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Creating basis functions

    ################################################## Basis functions ################################################
    column_list = X.columns
    for i in range(0, len(column_list)):
        j = i
        while (j < len(column_list)):
            temp = column_list[i] + str("_") + column_list[j]
            X[temp] = (X[column_list[i]] * X[column_list[j]]).astype(float)
            j = j + 1

    def create_basis_vectors(series, feature_name, power):

        for i in range(1, power + 1):
            temp = feature_name + str("_") + str(i)
            series[temp] = np.power(series[feature_name], i).astype(float)
        return series
    #
    X = create_basis_vectors(X, "latitude", 8)
    X = create_basis_vectors(X, "brightness", 8)
    X = create_basis_vectors(X, "bright_t31", 8)
    X = create_basis_vectors(X, "scan", 8)

    ####################################################################################################################

    # Feature scaling
    # is_train = True for train set, istrain = False for test set,dev set
    X = scaler(X, is_train)


    # Feature set now has Ones padded at index 0
    X.insert(0, "Ones", value=np.ones((X.shape[0], 1)))

    # converted to matrix form
    X = np.matrix(X)

    return X

def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    data_df = pd.read_csv(csv_path)

    # If it is a training set, outliers having frp above 3000 will be removed, for dev set, outliers above 9000 will be removed.
    # Train set will only be on those data points having frp value less than 3000 so that model doesn't have to work with outliers.
    #The above mentioned thing is done for the code on kaggle, here only "frp" is extracted.
    Y = data_df['frp']

    Y = np.matrix(Y).T  # Here transpose is taken because np.matrix converts Y of shape (250001,) into (1x250001)
    return Y






def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 5d
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    '''
    # Analytical solution calculated.

    # first indicates (X^T X)^(-1).
    i = np.ones((train_features.shape[1], train_features.shape[1]))
    first = np.linalg.pinv(np.matmul(np.transpose(train_features), train_features))

    # second indicates (X^T)Y.
    second = np.matmul(np.transpose(train_features), train_targets)

    # weights is multiplication of both the terms.
    weights = np.matmul(first, second)

    return weights


def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''
    y_pred =  np.matmul(feature_matrix,weights)
    return y_pred


def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''
    #diff is the difference matrix between targets and predicted.
    diff = targets - (get_predictions(feature_matrix, weights))

    #mse_loss = diff^2 / m
    #m = number of samples
    mse_loss = np.sum(np.power(diff,2))/diff.shape[0]

    return mse_loss


def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    #implementing the l2 regularization

    l2_Reg=np.sum(np.power(weights,2))
    return l2_Reg


def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''
    #implementing the loss function

    loss=mse_loss(feature_matrix,weights,targets)+C*l2_regularizer(weights)
    return loss


def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    m=feature_matrix.shape[0]
    A=np.dot(feature_matrix,weights)
    dw = (1 / m) * (np.dot(feature_matrix.T,(A - targets)) + C* weights)
    return dw

def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''
    #selecting the sample random batch based on the batch_size argument
    m = feature_matrix.shape[0]
    samples = np.random.randint(0,m,batch_size)
    sampled_features_matrix = feature_matrix[samples,:]
    sampled_targets = targets[samples,:]
    return (sampled_features_matrix, sampled_targets)


def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    '''
    Arguments
    n: int
    '''
    # initializing the weights to the standard normal values
    init_weights=np.random.randn(n,1)

    return init_weights


def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''
    #updating the weights, here lr--> learning rate
    weights = weights - lr*gradients
    return weights

def early_stopping(loss_dev_old, loss_dev_new):
    # allowed to modify argument list as per your need
    # return True or False
    #implementation of early stopping algorithm
    return (loss_dev_old<loss_dev_new)


def plot_trainsize_losses(train_features,
                        train_targets,
                        dev_feature_matrix,
                        dev_targets,
                        ):
    '''
    Description:
    plot losses on the development set instances as a function of training set size
    '''

    '''
    Arguments:
    # you are allowed to change the argument list any way you like 
    '''
    # function to plot the development loss as a function of train size
    dev_score = np.zeros(5)
    # step size= 5000
    train_size = np.array([5000, 10000, 15000, 20000, 25000])
    for i in range(1, 6):

        a_solution = analytical_solution(train_features[0:i * 5000, :],
                                             train_targets[0:i * 5000],
                                             C=1e8)

        dev_score[i - 1] = mse_loss(dev_feature_matrix[0:i * 5000, :], a_solution, dev_targets[0:i * 5000, :])


    plt.plot(train_size, dev_score)
    plt.xlabel('Train Size')
    plt.ylabel('Dev Score')
    plt.title('Dev Score vs Train Size')
    plt.grid()
    plt.show()
    return


def do_gradient_descent(train_feature_matrix,
                        train_targets,
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights

    a sample code is as follows --
    '''
    # function to perform gradient descent by calling the functions
    n = train_feature_matrix.shape[1]
    # initializing the weights
    weights = initialize_weights(n)
    # calculating the MSE loss
    dev_loss_new = mse_loss(dev_feature_matrix, weights, dev_targets)
    dev_loss_min = dev_loss_new
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)


    print("step {} \t dev loss: {} \t train loss: {}".format(0, dev_loss_new, train_loss))
    times = 0
    for step in range(1, max_steps + 1):

        # sample a batch of features and gradients
        features, targets = sample_random_batch(train_feature_matrix, train_targets, batch_size)

        # compute gradients
        gradients = compute_gradients(features, weights, targets, C)

        # update weights
        weights = update_weights(weights, gradients, lr)
        # ultimate_weights = weights

        dev_loss_new = mse_loss(dev_feature_matrix, weights, dev_targets)
        if (dev_loss_min > dev_loss_new):
            dev_loss_min = dev_loss_new
            ultimate_weights = weights
            None
        '''
        if (step%5==0):
          dev_loss_old = min(dev_loss_new,dev_loss_new)
          dev_loss_new = mse_loss(dev_feature_matrix, weights, dev_targets)
          #print(dev_loss_new,dev_loss_old)
          if (early_stopping(dev_loss_old, dev_loss_new)):
            ultimate_weight = weights_old
          else: 
            times = times+ 1

        if(times>10):
          if(pprint==True):
            print("***Early Stopping Gradient Descent***")
            return ultimate_weight
        '''
        if (step % eval_steps == 0):
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step, dev_loss, train_loss))

    return ultimate_weights


def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error
    predictions = get_predictions(feature_matrix, weights)
    loss = mse_loss(feature_matrix, weights, targets)
    return loss


if __name__ == '__main__':
    scaler = Scaler()  # use of scaler is optional
    train_features, train_targets = get_features('data/train.csv', True, scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv', False, scaler), get_targets('data/dev.csv')

    a_solution = analytical_solution(train_features, train_targets, C=1e8)
    print('evaluating analytical_solution...')
    dev_loss = do_evaluation(dev_features, dev_targets, a_solution)
    train_loss = do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    print('training LR using gradient descent...')

    gradient_descent_soln = do_gradient_descent(train_features,
                                                train_targets,
                                                dev_features,
                                                dev_targets,
                                                lr=0.01,
                                                C=1e-8,
                                                batch_size=512,
                                                max_steps=200000,
                                                eval_steps=20000,
                                                )

    dev_loss = do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss = do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('Gradient Descent \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    print('evaluating iterative_solution...')
    dev_loss = do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss = do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))


    plot_trainsize_losses(train_features,
                        train_targets,
                        dev_features,
                        dev_targets
                        )

