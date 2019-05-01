from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
import os

from collections import namedtuple
import pandas as pd
import numpy as np
import datetime
import pickle
from itertools import product
import pickle
from pyspark.sql.functions import *
import xlearn as xl
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
import os

appName = 'seven_mtimes'
master = 'local[*]'
dbName = 'seven_mtimes'
pqPath = '/seven_mtimes/temp'
modelPath = '/seven_mtimes/model'
users_path = r'/seven_mtimes/user.csv'
movies_path = r'/seven_mtimes/movie.csv'

spark = SparkSession.builder.appName(appName).master(master).config('spark.sql.catalogImplementation',
                                                                    'hive').getOrCreate()
spark.sql('create database if not exists %s' % dbName)
spark.sql('use %s' % dbName)


def show_table(table):
    res = spark.sql('select * from %s' % table)
    res.show()


def load_dict(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


def save_dict(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# Base Model
# MF + ALS

# the result of randomSplit is not stable, so use sklearn to split the dataset instead
# and remove all the sample in test dataset with uid not in train dataset, because spark ALS ignores cold start samples
def get_data():
    spark.sql('REFRESH TABLE rating')
    data = spark.sql('select id, uid, mid, rating from rating order by id asc')
    data_df = data.toPandas()
    train_df, test_df = train_test_split(data_df, random_state=0)
    train_uids = set(train_df['uid'].unique())
    train_mids = set(train_df['mid'].unique())
    print('test_df ', len(test_df))
    test_df = test_df[test_df.apply(lambda x: x['uid'] in train_uids and x['mid'] in train_mids, axis=1)]
    print('test_df ', len(test_df))
    return train_df, test_df


def test_spark_als(rank, maxiter, reg, train_data, test_data):
    print('test rank %d, epoch %d, lambda %f' % (rank, maxiter, reg))
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    als = ALS(rank=rank, maxIter=maxiter, regParam=reg, seed=0, userCol='uid', itemCol='mid', ratingCol='rating',
              coldStartStrategy="drop")
    model = als.fit(train_data)
    test_pred = model.transform(test_data)
    test_rmse = evaluator.evaluate(test_pred)
    return test_rmse


def test_mf(train_data, test_data):
    als_result_path = r'als_result.bin'
    als_result = load_dict(als_result_path)

    ranks = (10, 20, 35, 40, 55, 60, 70, 80, 85, 90, 95, 100, 110, 120)
    maxiters = (10, 20)
    regs = (0.01, .01, .015, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1)

    best_rank = 0
    best_miter = 0
    best_reg = 0
    best_rmse = float('inf')
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    for key, value in als_result.items():
        if value < best_rmse:
            best_rank, best_miter, best_reg = key
            best_rmse = value

    for rank, maxiter, reg in product(ranks, maxiters, regs):
        key = (rank, maxiter, reg)
        if key in als_result:
            continue
        als = ALS(rank=rank, maxIter=maxiter, regParam=reg, seed=0, userCol='uid', itemCol='mid', ratingCol='rating', \
                  coldStartStrategy="nan")
        model = als.fit(train_data)
        pred = model.transform(test_data)
        rmse = evaluator.evaluate(pred)
        if rmse < best_rmse:
            best_rank = rank
            best_miter = maxiter
            best_reg = reg
            best_rmse = rmse
        als_result[key] = rmse
        print('rank %d maxiter %d reg %f rmse %f' % (rank, maxiter, reg, rmse))
        with open(als_result_path, 'wb') as f:
            pickle.dump(als_result, f)

    print('best score: rank %d, maxIter %d, reg %f, rmse %f' % (best_rank, best_miter, best_reg, best_rmse))


# libFM
LIBFM_PATH = r'./libFM'
train_file = r'fm_train.dat'
test_file = r'fm_test.dat'
output_file = r'fm_output.dat'
model_file = 'fm_model.dat'


def lib_fm(train_file, test_file, trainY, testY, rank, reg, it):
    for file in (train_file, test_file, output_file):
        if os.path.exists(file):
            os.unlink(file)

    cmd = '%s -task r -method mcmc -train %s -test %s -iter %d -dim \'1,1,%d\' -out %s' % \
          (LIBFM_PATH, train_file, test_file, it, rank, output_file)
    os.system(cmd)
    pred = pd.read_csv(output_file, header=None).values.flatten()
    rmse = np.mean((testY - pred) ** 2) ** 0.5
    return rmse


def xl_fm(train_file, test_file, trainY, testY, rank, reg, epoch):
    for file in (model_file, output_file):
        if os.path.exists(file):
            os.unlink(file)

    param = {'task': 'reg', 'metric': 'rmse', 'epoch': epoch, 'k': rank, 'lambda': reg}
    fm_model = xl.create_fm()
    fm_model.setTrain(train_file)
    #     fm_model.setValidate(train_file)
    fm_model.fit(param, model_file)
    fm_model.setTest(test_file)
    fm_model.predict(model_file, output_file)
    pred = pd.read_csv(output_file, header=None).values.flatten()
    test_rmse = np.mean((testY - pred) ** 2) ** 0.5
    return test_rmse


def fm_grid_search(fm_func, result_path, ranks, maxiters, lambdas, trainX, trainY, testX, testY):
    best_iter = 0
    best_rank = 0
    best_lambda = 0
    best_rmse = float('inf')
    fm_result = load_dict(result_path)

    for key, value in fm_result.items():
        if value < best_rmse:
            best_rmse = value
            best_rank, best_lambda, best_iter = key
    if best_iter > 0:
        print('in cache: best rank %d lambda %f iter %d rmse %f' % (best_rank, best_lambda, best_iter, best_rmse))

    for file in (train_file, test_file):
        if os.path.exists(file):
            os.unlink(file)
    dump_svmlight_file(trainX, trainY, train_file)
    dump_svmlight_file(testX, testY, test_file)

    for rank, reg, epoch in product(ranks, lambdas, maxiters):
        key = (rank, reg, epoch)
        if key in fm_result:
            continue
        test_rmse = fm_func(train_file, test_file, trainY, testY, rank, reg, epoch)
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_rank = rank
            best_iter = epoch
            best_lambda = reg
        fm_result[key] = test_rmse
        print('rank %d lambda %f iter %d test rmse %f' % (rank, reg, epoch, test_rmse))
        save_dict(fm_result, result_path)

    print('best rank %d lambda %f iter %d rmse %f' % (best_rank, best_lambda, best_iter, best_rmse))


def test_xl_fm(train_df, test_df, result_path, ranks, epochs, lambdas):
    trainX = train_df.drop(['rating'], axis=1)
    trainY = train_df['rating']
    testX = test_df.drop(['rating'], axis=1)
    testY = test_df['rating']
    fm_grid_search(xl_fm, result_path, ranks, maxiters, lambdas, trainX, trainY, testX, testY)




if __name__ == '__main__':
    train_df, test_df = get_data()
    train_data, test_data = spark.createDataFrame(train_df), spark.createDataFrame(test_df)
    # test_mf(train_data, test_data)

    ranks = (8, 50, 64, 75, 128)
    maxiters = (10, 20, 30, 40, 50, 100)
    lambdas = (0.00002, .0005, .001, 0.002, 0.005, .01, .1)
    test_xl_fm(train_df, test_df, 'xl_fm_result.bin', ranks, maxiters, lambdas)
