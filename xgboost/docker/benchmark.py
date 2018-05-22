import numpy
import xgboost

# Datasets size
features_count = 100
train_length = int(1e5)
test_length = int(1e4)

# Model parameters
eta = 0.1
max_depth = 50
num_boost_rounds = 100

numpy.random.seed(0)

coefficients = numpy.random.rand(features_count, 1)

def generate_dataset(length):
    x = numpy.random.rand(length, features_count)
    t = x.dot(coefficients)
    median = numpy.median(t)
    y = t.copy()
    y[t <= median] = 0
    y[t > median] = 1
    return x, y

x_train, y_train = generate_dataset(train_length)
dtrain = xgboost.DMatrix(x_train, label=y_train)

x_test, y_test = generate_dataset(test_length)
dtest = xgboost.DMatrix(x_test, label=y_test)

model = xgboost.train(
    {
        'eta': eta,
        'max_depth': max_depth,
        'tree_method': 'exact',
        'eval_metric': 'auc',
        'silent': 1,
    },
    dtrain,
    evals=[(dtest, 'test')],
    num_boost_round=num_boost_rounds,
)
