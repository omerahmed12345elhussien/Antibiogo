"""
* tfrecord_loader.py for loading and preparing the TfRecord data during Training, Validating, and Testing.

"""
import tensorflow as tf
from utils_copy import AUTOTUNE,BUFFER_SIZE,train_dir,val_dir,test_dir,add_sample_weights
from tfrecord import parse_tfrecord_fn,prepare_sample
from os import getenv

if int(getenv("MODEL_OPTION"))<=2 or int(getenv("MODEL_OPTION"))==4 or (int(getenv("MODEL_OPTION"))==3 and 'MNet' in getenv("TAGS")):
    def parse_tfrecord_fn(example):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["image"] = -1+ tf.cast(tf.image.decode_png(example["image"], channels=3),tf.float32)/127.5
        example["mask"] = tf.squeeze(tf.image.decode_png(example["mask"],channels=1),axis=-1)
        return example
elif int(getenv("MODEL_OPTION"))==5 or (int(getenv("MODEL_OPTION"))==3 and 'Effi' in getenv("TAGS")):
    def parse_tfrecord_fn(example):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["image"] = tf.cast(tf.image.decode_png(example["image"], channels=3),tf.float32)
        example["mask"] = tf.squeeze(tf.image.decode_png(example["mask"],channels=1),axis=-1)
        return example

# Hyper-parameters
BATCH_SIZE = int(getenv("BATCH_SIZE"))
train_filenames = tf.io.gfile.glob(f"{train_dir}/*.tfrec")
vald_filenames = tf.io.gfile.glob(f"{val_dir}/*.tfrec")
#test_filenames = tf.io.gfile.glob(f"{test_dir}/*.tfrec")

single_batch = (
    tf.data.TFRecordDataset(vald_filenames, num_parallel_reads=AUTOTUNE)
    .shuffle(BATCH_SIZE*10)
    .take(1)
    .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    .map(prepare_sample, num_parallel_calls=AUTOTUNE)
    .batch(1)
    .prefetch(buffer_size=AUTOTUNE))

train_batches = (
    tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTOTUNE)
    .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    .map(prepare_sample, num_parallel_calls=AUTOTUNE)
    .shuffle(BATCH_SIZE*10,reshuffle_each_iteration=True)
    .batch(BATCH_SIZE,num_parallel_calls=AUTOTUNE)
    .prefetch(buffer_size=AUTOTUNE))

vald_batches = (
    tf.data.TFRecordDataset(vald_filenames, num_parallel_reads=AUTOTUNE)
    .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    .map(prepare_sample, num_parallel_calls=AUTOTUNE)
    .shuffle(BATCH_SIZE*10,reshuffle_each_iteration=True)
    .batch(BATCH_SIZE,num_parallel_calls=AUTOTUNE)
    .prefetch(buffer_size=AUTOTUNE))

# test_batches = (
#     tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTOTUNE)
#     .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
#     .map(prepare_sample, num_parallel_calls=AUTOTUNE)
#     .batch(BATCH_SIZE)
#     .prefetch(buffer_size=AUTOTUNE))
