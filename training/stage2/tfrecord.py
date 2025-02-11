"""
* tfrecord.py contains the tfrecord class that coverts the dataset to Tfrecord format.
We follow this implementation:
https://keras.io/examples/keras_recipes/creating_tfrecords/

"""
import tensorflow as tf
import matplotlib.pyplot as plt

class TfRecord():
    def __init__(self,dataset:tf.data.Dataset,tf_output_dir,num_samples:int=5120):
        self.dataset = dataset
        self.tf_output_dir = tf_output_dir
        self.num_samples = num_samples
        self.counter = 0
    
    global image_feature
    def image_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.image.encode_png(value).numpy()])
        )
        
    global create_example
    def create_example(image, mask):
        feature = {
            "image": image_feature(image),
            "mask": image_feature(mask),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    global prepare_sample
    def prepare_sample(features):
        return features["image"], features["mask"]
    
    global parse_tfrecord_fn
    def parse_tfrecord_fn(example):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["image"] = -1+ tf.cast(tf.image.decode_png(example["image"], channels=3),tf.float32)/127.5
        example["mask"] = tf.squeeze(tf.image.decode_png(example["mask"],channels=1),axis=-1)
        return example
    
    def generate_tf_record(self):
        data_size = tf.data.experimental.cardinality(self.dataset).numpy()
        num_tfrecords = data_size//self.num_samples
        if data_size%self.num_samples:
            num_tfrecords += 1  # add one record if there are any remaining samples
        skip_val = 0; take_val = self.num_samples if self.num_samples<data_size else data_size
        for tfrec_num in range(num_tfrecords):
            data_set = self.dataset.skip(skip_val).take(take_val)
            remaining_data = data_size-take_val
            with tf.io.TFRecordWriter(
                self.tf_output_dir + "/file_%.2i_%.3i-%i.tfrec" % (tfrec_num, self.counter, tf.data.experimental.cardinality(data_set).numpy())
            ) as writer:
                self.counter+=1
                skip_val = take_val;take_val=self.num_samples if remaining_data>self.num_samples else remaining_data
                for img,mask in data_set:
                    example = create_example(img,mask)
                    writer.write(example.SerializeToString())
    
