from pyspark.sql import *
from pyspark.ml.recommendation import ALSModel, ALS
from flask import session
import datetime

PAGE_SIZE = 20
appName = 'seven_mtimes'
master = 'local[*]'
dbName = 'seven_mtimes'
modelPath = '/seven_mtimes/als_model'

spark = SparkSession.builder.appName(appName).master(master).config('spark.sql.catalogImplementation', 'hive').getOrCreate()
spark.sql('use %s' % dbName)


def list_to_sql_range(values):
    string = ','.join((str(value) for value in values))
    return string.join(('(', ')'))


def get_name_list_from_ids(table, data):
    if len(data) == 0:
        return []
    sql = 'select id, name from %s where id in %s' % (table, list_to_sql_range(data))
    return spark.sql(sql).collect()


def get_movie_from_ids(mids):
    sql = 'select id, name, rating, actors, directors from movie where id in %s' % list_to_sql_range(mids)
    movies = spark.sql(sql).collect()
    return movies

def get_movie(mid):
    sql = 'select genres, actors, directors, district.name as district, rating, traits, movie.name as name \
    from movie, district where movie.id=%d and district.id = movie.district' % mid
    movies = spark.sql(sql).collect()
    movie = movies[0]
    actors = get_name_list_from_ids('actor', movie['actors'])
    directors = get_name_list_from_ids('director', movie['directors'])
    traits = get_name_list_from_ids('trait', movie['traits'])
    genres = get_name_list_from_ids('genre', movie['genres'])
    myrating = 0
    if 'user_id' in session:
        uid = session['user_id']
        sql = 'select rating from rating where uid=%d and mid=%d order by time desc limit 1'% (uid, mid)
        result = spark.sql(sql).collect()
        if len(result) == 1:
            myrating = result[0]['rating']
    result = Row(id=mid, name=movie.name, rating=movie.rating, district=movie.district, actors=actors,
                 directors=directors, traits=traits, genres=genres, myrating=myrating)
    return result


def get_name(table, nid):
    sql = 'select id, name, movies from %s where id = %d' % (table, nid)
    items = spark.sql(sql).collect()
    if len(items) == 0:
        return None
    item = items[0]
    mids = item['movies']
    sql = 'select id, name, rating from movie where id in %s  order by rating desc' % list_to_sql_range(mids)
    movies = spark.sql(sql).collect()
    return Row(item=item, movies=movies)


# here is the user-specific rating, not average rating
def get_user(uid):
    sql = 'select id, name, movies from user where id = %d' % uid
    # 'id', 'name'
    users = spark.sql(sql).collect()
    if len(users) == 0:
        return None
    user = users[0]
    ratings = get_user_ratings(uid)
    result = Row(user=user, ratings=ratings)
    return result


def get_user_ratings(uid):
    ratings = []
    sql = 'select distinct(mid) from rating where uid = %d' % uid
    result = spark.sql(sql).collect()
    mids = (row['mid'] for row in result)
    for mid in mids:
        sql = 'select mid as id, name as name, time, rating.rating as rating from rating, movie' \
              ' where uid=%d and mid=%d and mid=movie.id order by time desc limit 1' % (uid, mid)
        result = spark.sql(sql).collect()
        ratings.append(result[0])

    # 'rating', 'time', 'name', 'id'
    ratings.sort(reverse=True, key=lambda x: x['time'])
    return ratings



def get_actor(aid):
    return get_name('actor', aid)


def get_director(did):
    return get_name('director', did)


def get_movie_list(pid):
    sql = 'select id, name, rating from movie where id >= %d and id < %d order by id asc' % (pid * PAGE_SIZE, (pid+1) * PAGE_SIZE)
    movies = spark.sql(sql).collect()
    return movies


def get_name_list(table, pid):
    sql = 'select id, name from %s where id >= %d and id < %d order by id asc' % (table, pid * PAGE_SIZE, (pid+1) * PAGE_SIZE)
    return spark.sql(sql).collect()


def get_actor_list(pid):
    return get_name_list('actor', pid)


def get_director_list(pid):
    return get_name_list('director', pid)


def get_user_list(pid):
    return get_name_list('user', pid)


def select_random_users(count):
    sql = 'select user.name, user.id, rand() as random from user order by random limit %d' % count
    users = spark.sql(sql).collect()
    return users


def get_recall_dataset(uid):
    # sql = 'select id, actors, directors from movie, rating where rating.uid = %d and rating.mid = movie.id and rating.rating > 6' % uid
    # result = spark.sql(sql).collect()
    user_ratings = get_user_ratings(uid);
    rated_mids = set(item['id'] for item in user_ratings)
    related_mids = set(item['id'] for item in user_ratings if item['rating']>6)

    if len(related_mids) == 0:
        recall_mids = []
    else:
        actors = set()
        directors = set()
        result = get_movie_from_ids(related_mids)
        for movie in result:
            actors.update(movie.actors)
            directors.update(movie.directors)
        sql = 'select movies from actor where actor.id in ' + list_to_sql_range(actors)
        result1 = spark.sql(sql).collect()
        sql = 'select movies from director where director.id in ' + list_to_sql_range(directors)
        result2 = spark.sql(sql).collect()
        for result in (result1, result2):
            for record in result:
                related_mids.update(record.movies)

        related_mids -= (rated_mids)
        sql = 'select id from movie where id in %s and rating > 6 order by rating desc limit 50 ' % list_to_sql_range(related_mids)
        result = spark.sql(sql).collect()
        recall_mids = [record.id for record in result]

    if len(recall_mids) < 50:
        if len(recall_mids) > 0:
            sql = 'select id from movie where id not in %s order by rating desc limit %d ' % (list_to_sql_range(related_mids | (rated_mids)), 50 - len(recall_mids))
        else:
            sql = 'select id from movie where order by rating desc limit 50'
        result = spark.sql(sql).collect()
        recall_mids.extend((record.id for record in result))

    return recall_mids

def mark_rating(uid, mid, rating):
    count = spark.sql('select id from rating').count()
    sql = 'insert into rating values(%d, %d, %d, "%s", %d)' % (count, uid, mid, datetime.datetime.now(), rating)
    spark.sql(sql).collect()



class RecModel:
    def __init__(self):
        self.model = None
        try:
            self.model = ALSModel.load(modelPath)
        except Exception as e:
            print(e)
            self.train()

    def train(self):

        # best score: rank 95, maxIter 20, reg 0.600000, rmse 2.664016
        # (35, 10, 0.8) 2.6774639553502992
        rank = 35
        epoch = 10
        reg = 0.8
        als = ALS(rank=rank, maxIter=epoch, regParam=reg, seed=0, userCol='uid', itemCol='mid', ratingCol='rating',
                  coldStartStrategy="drop")
        data = spark.sql('select id, uid, mid, rating from rating order by id asc')
        self.model = als.fit(data)
        self.model.write().overwrite().save(modelPath)

    def predict(self, uid, mids, cnt):
        recall_mids = spark.createDataFrame([(uid, mid) for mid in mids], ["uid", "mid"])
        result = self.model.transform(recall_mids).collect()
        result.sort(reverse=True, key=lambda x: x['prediction'])
        result = result[:cnt]
        movies = get_movie_from_ids((item.mid for item in result))
        movies = {item.id: item for item in movies}
        result = [Row(id=item.mid, name=movies[item.mid].name, rating=movies[item.mid].rating) for item in result]
        # result = [Row(id=item.mid, rating=movies[item.mid].rating, pred=item.rating, name=movies[item.mid].name, district=movies[item.mid].district,
        #               actors=[actor.name for actor in movies[item.mid].actors],
        #               directors=[director.name for director in movies[item.mid].directors],
        #               traits=[trait.name for trait in movies[item.mid].traits],
        #               genres=[genre.name for genre in movies[item.mid].genres]) for item in rec_mids]
        return result
