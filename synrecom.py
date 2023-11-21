# -*- coding: utf-8 -*-
"""Synthesize Recommend Dataset"""

import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds

np.random.seed(1234)

user_batch_size = 1000
num_users = user_batch_size * 300
num_items = 100000
date_batch_size = timedelta(days=1)
start_date = datetime(2023, 10, 1)
end_date = datetime(2023, 11, 1)

gender_labels = ["alpha", "beta"]
location_labels = ["north", "east", "south", "west"]
color_labels = [
    "white", "black", "gray", "yellow", "red", "blue",
    "green", "brown", "pink", "orange", "purple"
]

user_location_probs = np.array([0.2, 0.25, 0.3, 0.25])
user_age_min = 0.0
user_age_max = 10.0
item_color_probs = np.array([
    14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 5
], np.float32) / 100
location_timedelta = np.array([0, 6, 12, 18]) * 60 * 60

hidden_dim = 16
gender_vec = np.random.randn(len(gender_labels), hidden_dim) * 1.0
location_vec = np.random.randn(len(location_labels), hidden_dim) * 0.5
age_vec = np.random.randn(3, hidden_dim) * 1.0
item_color_vec = np.random.randn(len(color_labels), hidden_dim) * 1.0
user_std = 0.3
session_std = 0.2
transaction_std = 0.1
item_std = 0.3

# Probability of ending session in one transaction
smooth = 5.0
session_start_probs_alpha = 1 * smooth
session_start_probs_beta = 3 * 24 * 60 * smooth
average_transactions_per_session = 6.0
average_transactions_timedelta = 5 * 60
transaction_prob = 0.3
session_end_prob = 0.03
time_of_day_weights = np.cos(np.linspace(0, 2 * np.pi, 24 * 60)) * 0.5 + 1


def make_users(num_users, location_probs, user_age_min, user_age_max):
    user_gender = np.random.choice([0, 1], size=num_users)
    user_location = np.random.choice(range(len(location_probs)), p=location_probs, size=num_users)
    user_age = np.random.uniform(low=user_age_min, high=user_age_max, size=num_users)
    return user_gender, user_age, user_location


def make_items(num_items, color_probs):
    item_color = np.random.choice(range(len(color_probs)), p=color_probs, size=num_items)
    return item_color,


def make_session(
    start_date, end_date,
    user_location, user_session_start_probs,
    time_of_day_weights
):
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    ts = np.arange(start_ts, end_ts, 60)
    x = np.random.rand(len(user_location), len(ts))
    x /= time_of_day_weights[None, (ts // 60) % (24 * 60)]
    x /= user_session_start_probs[:, None]
    session_user, session_ts_index = np.where(x < 1.0)
    session_ts = ts[session_ts_index] + np.random.randint(low=0, high=59, size=session_ts_index.shape)
    session_ts += location_timedelta[user_location[session_user]]
    return session_user, session_ts


def make_transaction(session_user, session_ts):
    transaction_session = []
    transaction_ts = []
    num_transactions = np.random.poisson(average_transactions_per_session - 1, size=session_user.shape) + 1
    for s, (u, ts, nt) in enumerate(zip(session_user, session_ts, num_transactions)):
        timedelta = np.cumsum(np.random.poisson(average_transactions_timedelta - 1, nt - 1) + 1)
        transaction_session.append(np.array([s] * nt))
        transaction_ts.append(np.concatenate([ts[None], ts + timedelta]))
    transaction_session = np.concatenate(transaction_session)
    transaction_ts = np.concatenate(transaction_ts)
    return transaction_session, transaction_ts


def make_user_vec(gender, age, location, user_age_min, user_age_max):
    age_t = (age - user_age_min) / user_age_max
    vec = gender_vec[gender] + location_vec[location]
    vec += spline(age_t, age_vec)
    vec += np.random.randn(*vec.shape) * user_std
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec


def make_item_vec(color):
    vec = item_color_vec[color].copy()
    vec += np.random.randn(*vec.shape) * item_std
    return vec


def make_session_vec(user_vec, session_user):
    vec = user_vec[session_user].copy()
    vec += np.random.randn(*vec.shape) * session_std
    return vec


def make_transaction_vec(session_vec, transaction_session):
    transaction_vec = session_vec[transaction_session].copy()
    transaction_vec += np.random.randn(*transaction_vec.shape) * transaction_std
    return transaction_vec


def spline(t, cp):
    t = t[:, None]
    u = 1 - t
    cp = np.split(age_vec, 3)
    cp = [
        t * cp[0] + u * cp[1],
        t * cp[1] + u * cp[2]
    ]
    return t * cp[0] + u * cp[1]


def write_item(item_part_index, item_id, item_color):
    color_dict = pa.dictionary(pa.int8(), pa.utf8())
    table = pa.table(
        [
            pa.array(item_id),
            pa.array([color_labels[x] for x in item_color], color_dict),
        ],
        ['item', 'color'])
    pq.write_table(table, f'tmp/item-{item_part_index:04d}.parquet')


def write_user(user_part_index, user_id, user_gender, user_age, user_location):
    table = pa.table(
        [
            pa.array(user_id),
            pa.array([gender_labels[x] for x in user_gender]),
            pa.array(user_age, pa.float32()),
            pa.array([location_labels[x] for x in user_location]),
        ],
        ['user', 'gender', 'age', 'location'])
    pq.write_table(table, f'tmp/user-{user_part_index:04d}.parquet')


def write_transaction(
    date_part,
    user_part_index,
    transaction_user,
    user_id,
    transaction_session_id,
    transaction_ts, transaction_item, item_id
):
    table = pa.table(
        [
            pa.array([user_id[x] for x in transaction_user]),
            pa.array(transaction_session_id),
            pc.cast(transaction_ts, pa.timestamp('s')),
            pa.array([item_id[x] for x in transaction_item]),
        ],
        ['user', 'session', 'timestamp', 'item'])
    pq.write_table(table, f'tmp/transaction-{date_part}-{user_part_index:04d}.parquet')


def make_random_id(n, prefix, nbytes):
    x = np.random.randint(low=0, high=255, size=[n, nbytes], dtype=np.uint8)
    f = prefix + '%02x' * nbytes
    return [
        f % tuple(x[i, :].tolist())
        for i in range(n)
    ]


def generate():

    item_part_index = 0
    user_part_index = 0

    item_color, = make_items(
        num_items=num_items,
        color_probs=item_color_probs)
    item_id = make_random_id(num_items, 'i', 8)
    item_color, = make_items(
        num_items=num_items,
        color_probs=item_color_probs)
    write_item(item_part_index, item_id, item_color)

    print()

    for _ in range(0, num_users, user_batch_size):

        user_id = make_random_id(user_batch_size, 'u', 12)
        user_gender, user_age, user_location = make_users(
            num_users=user_batch_size,
            user_age_min=user_age_min,
            user_age_max=user_age_max,
            location_probs=user_location_probs)
        write_user(
            user_part_index, user_id, user_gender, user_age, user_location)

        # Probablity of starting session in one minutes
        user_session_start_probs = np.random.beta(
            session_start_probs_alpha,
            session_start_probs_beta,
            size=user_batch_size)

        user_vec = make_user_vec(
            user_gender, user_age, user_location,
            user_age_min, user_age_max)
        item_vec = make_item_vec(item_color)

        date = start_date - timedelta(days=1)
        while date < end_date:
            date_part = date.strftime('%Y-%m-%d')
            print(f'\r{date_part}-{user_part_index:04d}', end='')
            session_user, session_ts = make_session(
                date, date + date_batch_size,
                user_location, user_session_start_probs,
                time_of_day_weights)
            transaction_session, transaction_ts = make_transaction(
                session_user, session_ts)

            session_vec = make_session_vec(
                user_vec, session_user)
            transaction_vec = make_transaction_vec(
                session_vec, transaction_session)

            session_id = make_random_id(len(session_user), 's', 6)
            transaction_item = np.argmax(transaction_vec @ item_vec.T, axis=-1)
            transaction_session_id = [session_id[x] for x in transaction_session]
            transaction_user = [session_user[x] for x in transaction_session]
            write_transaction(
                date_part, user_part_index, transaction_user, user_id,
                transaction_session_id,
                transaction_ts, transaction_item, item_id)

            date += date_batch_size
        user_part_index += 1
    item_part_index += 1


def combine():
    rootdir = Path('tmp')

    files = list(rootdir.glob('user-*.parquet'))
    dataset = ds.dataset(files)
    table = dataset.to_table()
    pq.write_table(table, 'output/user.parquet')

    files = list(rootdir.glob('item-*.parquet'))
    dataset = ds.dataset(files)
    table = dataset.to_table()
    pq.write_table(table, 'output/item.parquet')

    files = list(rootdir.glob('transaction-*.parquet'))
    dataset = ds.dataset(files)
    table = dataset.to_table(
        filter=(ds.field('timestamp') >= start_date) & (ds.field('timestamp') < end_date))
    pq.write_table(table, 'output/transaction.parquet')


def main():
    os.makedirs('tmp', exist_ok=False)
    os.makedirs('output', exist_ok=False)
    generate()
    combine()


if __name__ == '__main__':
    main()
