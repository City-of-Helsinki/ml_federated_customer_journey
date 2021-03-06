# AUTOGENERATED! DO NOT EDIT! File to edit: 00_customclientdata.ipynb (unless otherwise specified).

__all__ = ['create_tff_client_data_from_df']

# Cell
import nest_asyncio

nest_asyncio.apply()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

# Cell

import collections


def create_tff_client_data_from_df(
    df,
    client_id_col="client_id",
    sample_size=1.0,
    shuffle_buffer=100,
    batch_size=32,
    num_epochs=20,
    prefetch_buffer=100,
    shuffle_seed=42,
):
    """
    turn pd dataframe into tff client dataset
    """

    def batch_format_fn(element):
        """format data into OrderedDict where x denotes features and y labels for a client"""
        return collections.OrderedDict(
            x=element["x"],  # tf.reshape(element[xcol], [-1, xshape]),
            y=element["y"],  # tf.reshape(element[ycol], [-1, yshape]),
        )

    def create_tf_dataset_for_client_fn(client_id):
        """a function which takes a client_id and returns a tf.data.Dataset for that client"""
        client_data = df[df[client_id_col] == int(client_id)]
        # create tf dataset
        dataset = tf.data.Dataset.from_tensor_slices(client_data.to_dict("list"))
        # dataset = dataset.shuffle(shuffle_buffer).batch(num_batch).repeat(num_epochs)
        dataset = (
            dataset.repeat(num_epochs)
            .shuffle(shuffle_buffer, seed=shuffle_seed)
            .batch(batch_size)
            .map(batch_format_fn)
            .prefetch(prefetch_buffer)
        )
        return dataset

    # split client id into train and test clients
    client_ids = np.random.choice(
        df[client_id_col].unique(),
        size=int(sample_size * df[client_id_col].nunique()),
    ).tolist()  # proportion of clients to use

    # train data
    def create_mapping(client_ids):
        """
        create mapping of client ids
        """
        mapping = {}
        for client_id in client_ids:
            mapping[client_id] = str(client_id)
        return mapping

    # create CustomClientData
    # this is slight misuse of FilePerUserClientData but it works for now,
    # and there seems not to be another solution for the moment :D
    # (please suggest correction if there is a better solution)
    tff_data = tff.simulation.datasets.FilePerUserClientData(
        client_ids_to_files=create_mapping(client_ids),
        dataset_fn=create_tf_dataset_for_client_fn,
    )

    return tff_data