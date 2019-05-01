# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:52:41 2019

@author: Leon
"""
import os
from random import randint
from flask import render_template, url_for, redirect
from flask import Flask, session, request, flash
from pyspark.sql import Row
import db
import random

app = Flask(__name__)
app.secret_key = os.urandom(32)
model = db.RecModel()


def img_path():
    return '/static/pic%d.jpg' % randint(0, 2)


@app.before_request
def before_request():
    if 'user_name' not in session:
        # user = db.select_random_users(1)[0]
        user = db.get_user(0).user
        if 'user_name' not in session:
            flash('你尚未选择用户身份，现在系统选择一个用户登录，你可以通过“切换用户”选择新的用户身份')
            session['user_name'] = user.name
            session['user_id'] = user.id
        return redirect(url_for('change_role', role=user.id))



@app.route('/')
def index():
    movies = db.get_movie_list(0)[:4]
    actors = db.get_actor_list(0)[:4]
    directors = db.get_director_list(0)[:4]
    return render_template('index.html',movies=movies, actors=actors, directors=directors)


@app.route('/movie/<int:mid>')
def movie(mid):
    result = db.get_movie(mid)
    if result is None:
        return redirect(url_for('index'))
    return render_template('movie.html', result=result)


@app.route('/actor/<int:aid>')
def actor(aid):
    result = db.get_actor(aid)
    if result is None:
        return redirect(url_for('index'))
    return render_template('name.html', result=result, name='演员', title='参演作品')


@app.route('/director/<int:did>')
def director(did):
    result = db.get_director(did)
    if result is None:
        return redirect(url_for('index'))
    return render_template('name.html', result=result, name='导演', title='导演作品')


@app.route('/user/<int:uid>')
def user(uid):
    result = db.get_user(uid)
    if result is None:
        return redirect(url_for('index'))
    return render_template('user.html', result=result)


@app.route('/movies')
@app.route('/movies/<int:pid>')
def movie_list(pid=0):
    movies = db.get_movie_list(pid)
    return render_template('movies.html', movies=movies, pid=pid, page_size=db.PAGE_SIZE)


@app.route('/actors')
@app.route('/actors/<int:pid>')
def actor_list(pid=0):
    actors = db.get_actor_list(pid)
    return render_template('names.html', items=actors, pid=pid, name = '演员', link_name1 ='actor', link_name2='actors',\
                           page_size=db.PAGE_SIZE)


@app.route('/directors')
@app.route('/directors/<int:pid>')
def director_list(pid=0):
    directors = db.get_director_list(pid)
    return render_template('names.html', items=directors, pid=pid, name='导演', link_name1='director', \
                           link_name2='directors', page_size=db.PAGE_SIZE)


@app.route('/users')
@app.route('/users/<int:pid>')
def user_list(pid=0):
    users = db.get_user_list(pid)
    return render_template('names.html', items=users, pid=pid, name='用户', link_name1='user', \
                           link_name2='users', page_size=db.PAGE_SIZE)


@app.route('/recommend')
def recommend():
    return 'recommend'


@app.route('/change_role')
def change_role():
    try:
        role = int(request.values.get('role'))
    except:
        return redirect(url_for('index'))

    result = db.get_user(role)
    if result is None:
        return redirect(url_for('index'))
    recall_mids = db.get_recall_dataset(role)
    session['user_id'] = result.user.id
    session['user_name'] = result.user.name
    session['recall_mids'] = recall_mids
    session['recom_movies'] = model.predict(result.user.id, recall_mids, 3)
    return redirect(url_for('user', uid=result.user.id))


@app.route('/rating')
def rating():
    try:
        mid = int(request.values.get('mid'))
        rating = int(request.values.get('rating'))
    except:
        return redirect(url_for('index'))
    if 'user_id' not in session:
        return redirect(url_for('index'))
    uid = session['user_id']
    db.mark_rating(uid, mid, rating)
    recall_mids = db.get_recall_dataset(uid)
    session['recall_mids'] = recall_mids
    session['recom_movies'] = model.predict(uid, recall_mids, 3)
    return redirect(url_for('movie', mid=mid))


def get_recom_movies():
    return [Row(id=item[0], name=item[1], rating=item[2]) for item in session['recom_movies']]


if __name__ == '__main__':
    app.jinja_env.globals.update(img_path=img_path, select_random_users=db.select_random_users, get_recom_movies=get_recom_movies)
    app.run(host='0.0.0.0', port=12345)