#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import monotonically_increasing_id
import os
from collections import namedtuple
import pandas as pd
import numpy as np
import datetime
import pickle
from itertools import product
import collections

appName = 'seven_mtimes'
master = 'local[*]'
dbName = 'seven_mtimes'
pqPath = '/seven_mtimes/temp'
modelPath = '/seven_mtimes/model'
movie_path = r'../data/movie.csv'
user_path = r'../data/user.csv'

spark = SparkSession.builder.appName(appName).master(master).config('spark.sql.catalogImplementation',
                                                                    'hive').getOrCreate()
spark.sql('create database if not exists %s' % dbName)
spark.sql('use %s' % dbName)

Movie = namedtuple('Movie', ['id', 'name', 'genres', 'actors', 'district', 'directors', 'traits', 'rating'])
Name = namedtuple('Name', ['id', 'name', 'movies'])
Rating = namedtuple('Rating', ['uid', 'mid', 'time', 'rating'])


def show_table(table):
    res = spark.sql('select * from %s' % table)
    res.show()


def put_to_hive(df, table, cols):
    df = spark.createDataFrame(df)
    tmp = os.path.join(pqPath, table)
    df.write.mode('overwrite').parquet(tmp)
    spark.sql("drop table if exists %s" % table)
    spark.sql("create table if not exists %s(%s) stored as parquet" % (table, cols))
    spark.sql("load data inpath '%s' overwrite into table %s" % (tmp, table))


# used to actor/director/genre/trait/district collection
# return the id and put the new name in the dict
def get_name_id(name_map, name, mid):
    if name in name_map:
        name_item = name_map[name]
        name_item.movies.add(mid)
        nid = name_item.id
    else:
        nid = len(name_map)
        name_item = Name(nid, name, set([mid]))
        name_map[name] = name_item
    return nid


# movie_df contains redundant records
# 剧情,徐峥|王传君|周一围|谭卓|章宇,中国大陆,文牧野,经典,8.9,我不是药神
# 剧情,徐峥|王传君|周一围|谭卓|章宇,中国大陆,文牧野,感人,8.9,我不是药神
# 喜剧,徐峥|王传君|周一围|谭卓|章宇,中国大陆,文牧野,经典,9.0,我不是药神
# 喜剧,徐峥|王传君|周一围|谭卓|章宇,中国大陆,文牧野,搞笑,9.0,我不是药神
# 喜剧,徐峥|王传君|周一围|谭卓|章宇,中国大陆,文牧野,感人,9.0,我不是药神
# 犯罪,徐峥|王传君|周一围|谭卓|章宇,中国大陆,文牧野,感人,9.0,我不是药神
# 犯罪,徐峥|王传君|周一围|谭卓|章宇,中国大陆,文牧野,搞笑,9.0,我不是药神
def process_movie_df():
    movie_df = pd.read_csv(movie_path, header=0,
                           names=['genre', 'actor', 'district', 'director', 'trait', 'rating', 'name']);
    movie_map = collections.OrderedDict()
    genre_map = collections.OrderedDict()
    actor_map = collections.OrderedDict()
    trait_map = collections.OrderedDict()
    director_map = collections.OrderedDict()
    district_map = collections.OrderedDict()

    # genre  actors  district  directors trait  rating  name
    for _, row in movie_df.iterrows():
        genre, actors, district, directors, trait, rating, movie_name = row
        if movie_name in movie_map:
            movie_record = movie_map[movie_name]
            mid = movie_record.id
        else:
            mid = len(movie_map)
            district_id = get_name_id(district_map, district, mid)
            actors = [get_name_id(actor_map, actor_name, mid) for actor_name in actors.split('|')]
            directors = [get_name_id(director_map, director_name, mid) for director_name in directors.split('|')]
            movie_record = Movie(mid, movie_name, set(), actors, district_id, directors, set(), rating)
            movie_map[movie_name] = movie_record
        genre_id = get_name_id(genre_map, genre, mid)
        trait_id = get_name_id(trait_map, trait, mid)
        movie_record.genres.add(genre_id)
        movie_record.traits.add(trait_id)

    rating_list = [row.rating for row in movie_map.values()]
    maps = (movie_map, trait_map, genre_map, actor_map, district_map, director_map)

    dfs = [pd.DataFrame.from_dict(df_map, orient='index').reset_index(drop=True) for df_map in maps]
    movie_df = dfs[0]
    movie_df.loc[:, ['genres', 'traits']] = movie_df[['genres', 'traits']].applymap(lambda x: list(x))
    for df in dfs[1:]:
        df['movies'] = df['movies'].apply(lambda x: sorted(list(x), key=lambda mid: rating_list[mid], reverse=True))
    return dfs


# save to hive
def movie_df_to_hive(movie_df, trait_df, genre_df, actor_df, district_df, director_df):
    params = ((movie_df, 'movie',
               'id bigint, name string, genres array<bigint>, actors array<bigint>, district bigint, directors array<bigint>, traits array<bigint>, rating double'),
              (actor_df, 'actor', 'id bigint, name string, movies array<bigint>'),
              (director_df, 'director', 'id bigint, name string, movies array<bigint>'),
              (genre_df, 'genre', 'id bigint, name string, movies array<bigint>'),
              (district_df, 'district', 'id bigint, name string, movies array<bigint>'),
              (trait_df, 'trait', 'id bigint, name string, movies array<bigint>')
              )
    for param in params:
        df = param[0]
        put_to_hive(*param)


def process_rating_df():
    rating_df = pd.read_csv(user_path, header=0, index_col=False, names=['rating', 'name', 'time', 'uid', 'movie'],
                            parse_dates=['time'])
    movie_df = spark.sql('select id, name, rating from movie').toPandas()
    movie_map = {row['name']: row['id'] for _, row in movie_df.iterrows()}
    rating_list = movie_df['rating'].tolist()
    user_map = collections.OrderedDict()
    print('csv rows ', len(rating_df))
    rating_map = collections.OrderedDict()
    for _, row in rating_df.iterrows():
        rating, user_name, time, uid, movie_name = row

        mid = movie_map[movie_name]
        uid = get_name_id(user_map, user_name, mid)
        rating_key = (uid, mid)
        if rating_key in rating_map:
            continue
        rating_record = Rating(uid, mid, time, rating)
        rating_map[rating_key] = rating_record

    rating_df = pd.DataFrame.from_dict(rating_map, orient='index').reset_index(drop=True)
    user_df = pd.DataFrame.from_dict(user_map, orient='index').reset_index(drop=True)
    user_df['movies'] = user_df['movies'].apply(
        lambda x: sorted(list(x), key=lambda mid: rating_list[mid], reverse=True))
    for df in (user_df, rating_df):
        df['id'] = df.index
    return rating_df, user_df


def user_df_to_hive(rating_df, user_df):
    params = ((rating_df, 'rating', 'id bigint, uid bigint, mid bigint, time timestamp, rating bigint'),
              (user_df, 'user', 'id bigint, name string, movies array<bigint>'))
    for param in params:
        put_to_hive(*param)


if __name__ == '__main__':
    movie_df, trait_df, genre_df, actor_df, district_df, director_df = process_movie_df()
    movie_df_to_hive(movie_df, trait_df, genre_df, actor_df, district_df, director_df)
    rating_df, user_df = process_rating_df()
    user_df_to_hive(rating_df, user_df)
    print('movies %d, director %d, actors %d, users %d, ratings %d' %
          (len(movie_df), len(director_df), len(actor_df), len(user_df), len(rating_df)))
