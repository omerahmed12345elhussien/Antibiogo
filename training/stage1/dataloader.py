"""
* dataloader.py for loading and preparing the data during Training, Validating, and Testing.

"""
import tensorflow as tf
from utils import AUTOTUNE,BUFFER_SIZE,train_dir,val_dir,test_dir,squeeze_mask

# Hyper-parameters
BATCH_SIZE = 1

train_dataset = tf.data.Dataset.load(train_dir)
val_ds = tf.data.Dataset.load(val_dir)
#test_ds = tf.data.Dataset.load(test_dir)

single_batch = (
    val_ds
    .shuffle(BUFFER_SIZE)
    .take(1)
    .cache()
    .batch(1)
    .map(squeeze_mask)
    .prefetch(buffer_size=AUTOTUNE))
    
train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    .batch(BATCH_SIZE,num_parallel_calls=AUTOTUNE)
    .map(squeeze_mask)
    .prefetch(buffer_size=AUTOTUNE))

vald_batches = val_ds.batch(BATCH_SIZE).map(squeeze_mask)

#test_batches = test_ds.batch(BATCH_SIZE).map(squeeze_mask)
