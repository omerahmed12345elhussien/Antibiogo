"""
* utils.py contains variables and functions that are  needed during debugging and training phases.
For computing Flops,we are following these two implementations:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/python_api.md
https://stackoverflow.com/questions/45085938/tensorflow-is-there-a-way-to-measure-flops-for-a-model

"""
import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img
from os import path,getenv,listdir
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import cv2 as cv

# For the cluster
#root_path = getenv("ROOT_DIR")
# For the local machine
root_path = "add_your_root_directory"
data_pth = path.join(root_path,"Data_stage1")
train_dir = path.join(root_path,"saved_data/Train")
val_dir = path.join(root_path,"saved_data/Valid")
test_dir = path.join(root_path,"saved_data/Test")
orig_train_dir = path.join(root_path,"saved_data/Original_Train")
tf_global_seed = 1234
np_seed = 1234
shuffle_data_seed = 12345
# Hyper-parameters
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1024
# The required image size.
IMG_SIZE = 256
# Number of classes in the output mask.
OUTPUT_CLASSES = 2

def display(display_list:list)->None:
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

def create_mask(pred_mask:tf.Tensor)->tf.Tensor:
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

# Instantiate an optimizer.
optimAdam = tf.keras.optimizers.Adam(learning_rate=0.0003)

def serve_model(model_path:str):
  """A function that adds the preprocessing layers (resizing, rescaling) to the trained model, 
  then it coverts it to TfLite model.

  Args:
      model_path (str): A path to the .keras model.

  Returns:
      _type_: A TfLite model.
  """
  inputs = tf.keras.layers.Input(shape=[None, None, 3])
  # Resize the input image to IMG_SIZE,IMG_SIZE,3]
  layer1 = tf.keras.layers.Resizing(height=IMG_SIZE,width=IMG_SIZE)(inputs)
  # Rescale the image to [-1,1]
  layer2 = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)(layer1)
  # Load the model.keras
  trained_model = tf.keras.models.load_model(model_path,compile=False)(layer2)
  preprocess_model = tf.keras.Model(inputs=inputs,outputs=trained_model)
  # Convert the model.
  converter = tf.lite.TFLiteConverter.from_keras_model(preprocess_model)
  tflite_model = converter.convert()
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  # To optimize inference
  interpreter.allocate_tensors()  # Needed before execution!
  classify_lite = interpreter.get_signature_runner('serving_default')
  return classify_lite

def model_flop(model_path):
  model = tf.keras.models.load_model(model_path)
  forward_pass = tf.function(model.call,
                             input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
  graph_info = profile(forward_pass.get_concrete_function().graph,
                       options=ProfileOptionBuilder.float_operation())
  flops = graph_info.total_float_ops // 2
  return round(flops/1e9,2)

def prepare_data(imgs_path:str)->tf.data.Dataset:
  imgs_path = Path(imgs_path).with_suffix('')
  # A dataset of all files in imgs_path folder.
  list_ds = tf.data.Dataset.list_files([str(imgs_path/'*.jp*'),str(imgs_path/'*.JP*')], shuffle=False)
  # Load the raw data from the file as a string.  
  img_ds = list_ds.map(tf.io.read_file)
  # Covert the compressed string to a 3D uint8 tensor.
  img_ds=img_ds.map(tf.io.decode_jpeg)
  img_ds=img_ds.map(lambda x: tf.image.resize(x,(IMG_SIZE,IMG_SIZE)))
  img_ds=img_ds.map(lambda x:-1 + tf.cast(x, tf.float32) / 127.5 )
  img_ds = img_ds.batch(1)
  return img_ds

def predict_newimgs(img_path,model_path,output_path_mask,stand_img=True,img_255=True):
  """Predict masks using the given image.

  Args:
      img_path (_type_): Image path.
      model_path (_type_): Model path.
      output_path_mask (_type_): Output directory to save the generated masks.
      stand_img (bool, optional): standardize the image values to be in the range [-1,1]. Defaults to True.
      img_255 (bool, optional): Save the image with values in the range [0,255]. Defaults to True.
  """
  model = tf.keras.models.load_model(model_path,compile=True)
  img_files = listdir(img_path)
  if ".DS_Store" in img_files: img_files.remove(".DS_Store")
  for img in img_files:
    _, ext = path.splitext(img)
    img_fullpath = path.join(img_path,img)
    load_img = tf.keras.utils.load_img(img_fullpath)
    img_height,img_width=load_img.height,load_img.width
    load_img = tf.keras.utils.load_img(img_fullpath,target_size=(IMG_SIZE, IMG_SIZE))
    input_img = tf.keras.utils.img_to_array(load_img)
    load_img = -1+tf.cast(input_img.copy(), tf.float32) / 127.5 if stand_img else tf.cast(input_img.copy(), tf.float32)
    load_img = tf.expand_dims(load_img, 0) # Create a batch
    model_prediction=create_mask(model.predict(load_img,verbose=0)) #(256, 256, 1)
    array_to_img(model_prediction).save(path.join(output_path_mask,f"{_}.png")) if img_255 else cv.imwrite(path.join(output_path_mask,f"{_}.png")
                                                                                                           ,tf.squeeze(model_prediction,axis=-1)
                                                                                                           )

def predict_imgs_with_padding(img_path,model_path,output_path_mask,stand_img=True,img_255=True):
  """Predict masks using the given image while keeping the aspect ratio of the image.

  Args:
      img_path (_type_): Image path.
      model_path (_type_): Model path.
      output_path_mask (_type_): Output directory to save the generated masks.
      stand_img (bool, optional): standardize the image values to be in the range [-1,1]. Defaults to True.
      img_255 (bool, optional): Save the image with values in the range [0,255]. Defaults to True.
  """
  model = tf.keras.models.load_model(model_path,compile=True)
  img_files = listdir(img_path)
  if ".DS_Store" in img_files: img_files.remove(".DS_Store")
  for img in img_files:
    _, ext = path.splitext(img)
    img_fullpath = path.join(img_path,img)
    load_img = tf.io.read_file(img_fullpath)
    load_img = tf.io.decode_png(load_img, channels=3)
    load_img = tf.image.resize_with_pad(load_img, target_height=IMG_SIZE, target_width=IMG_SIZE,method='bicubic')
    input_img = tf.keras.utils.img_to_array(load_img)
    load_img = -1+tf.cast(input_img.copy(), tf.float32) / 127.5 if stand_img else tf.cast(input_img.copy(), tf.float32)
    load_img = tf.expand_dims(load_img, 0) # Create a batch
    model_prediction=create_mask(model.predict(load_img,verbose=0)) #(256, 256, 1)
    array_to_img(model_prediction).save(path.join(output_path_mask,f"{_}.png")) if img_255 else cv.imwrite(path.join(output_path_mask,f"{_}.png")
                                                                                                           ,tf.squeeze(model_prediction,axis=-1)
                                                                                                           ) 
    
def predict_datadist(model_path,dataset_path,outputfile_path,output_name):
  model = tf.keras.models.load_model(model_path)
  val_data = tf.data.Dataset.load(dataset_path)
  batched_dataset = val_data.batch(1).map(squeeze_mask)
  results = []
  for img,mask in tqdm(batched_dataset):
    results_list = model.evaluate(x=img,y=mask,verbose=0)
    results.append(results_list[1])
  data_dict = {'mIOU':results}
  results= None
  df = pd.DataFrame(data_dict)
  df.to_csv(path.join(outputfile_path,f'Output_{output_name}.csv'))
  data_dict,df = None,None
                