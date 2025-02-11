"""
* utils_copy.py contains variables and functions that are may assist during debugging and training phase.

"""
from typing import List
import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img
from os import path,getenv,listdir
import tensorflow as tf

out_folder_name ="saved_data_copy_st2"
root_path = getenv("ROOT_DIR_ST2")
data_pth = path.join(root_path,"Data512_copy_st1st2")#
train_dir = path.join(root_path,out_folder_name,"Train")
val_dir = path.join(root_path,out_folder_name,"Valid")
test_dir = path.join(root_path,out_folder_name,"Test")
orig_train_dir = path.join(root_path,out_folder_name,"Original_Train")
tf_global_seed = 1234
np_seed = 1234
shuffle_data_seed = 12345
# Hyper-parameters
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1024
# The required image size.
IMG_SIZE = 512#256
OUTPUT_CLASSES = 3


def display(display_list:List)->None:
  """A function to display the true image, the true mask, and the predicted mask.

  Args:
      display_list (List): A list contains the [true_image,true_mask,predicted_mask].
  """
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for idx in range(len(display_list)):
    plt.subplot(1, len(display_list), idx+1)
    plt.title(title[idx])
    plt.imshow(array_to_img(display_list[idx]))
    plt.axis('off')
  plt.show()
  
def squeeze_mask(img,mask):
        return img,tf.squeeze(mask,axis=-1)

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def add_sample_weights(image, mask):
    """https://www.tensorflow.org/tutorials/images/segmentation#optional_imbalanced_classes_and_class_weights"""
    # The weights for each class, with the constraint that:
    #     sum(class_weights) == 1.0
    class_weights = tf.constant([1, 3, 6])
    class_weights = class_weights/tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(mask, tf.int32))

    return image, mask, sample_weights
  
# Instantiate an optimizer.
optimAdam = tf.keras.optimizers.Adam(learning_rate=0.0003,weight_decay=float(getenv("Weight_Decay")))

def predict_newimgs(img_path,model_path,output_path):
  model = tf.keras.models.load_model(model_path,compile=True)
  img_files = listdir(img_path)
  if ".DS_Store" in img_files: img_files.remove(".DS_Store")
  for img in img_files:
    img_fullpath = path.join(img_path,img)
    load_img = tf.keras.utils.load_img(img_fullpath)#, target_size=(IMG_SIZE, IMG_SIZE))
    img_height,img_width=load_img.height,load_img.width
    load_img = tf.keras.utils.load_img(img_fullpath,target_size=(IMG_SIZE, IMG_SIZE))
    load_img = tf.keras.utils.img_to_array(load_img)
    load_img = -1+tf.cast(load_img, tf.float32) / 127.5
    load_img = tf.expand_dims(load_img, 0) # Create a batch
    array_to_img(create_mask(model.predict(load_img))).resize(size=(img_width, img_height)).save(path.join(output_path,img))
