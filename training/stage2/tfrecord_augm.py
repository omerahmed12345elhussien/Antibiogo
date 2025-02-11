"""
* tfrecord_augm.py contains the data augmentation pipeline for the datast using Tfrecorder approach.

"""

from numpy.random import seed as seednp
import tensorflow as tf
import tensorflow_io as tfio
from typing import Tuple
from pathlib import Path
from scipy.ndimage import rotate as ndimage_rotate
from os import makedirs
from utils_copy import IMG_SIZE,BUFFER_SIZE,AUTOTUNE,shuffle_data_seed,tf_global_seed,np_seed,data_pth,train_dir,val_dir,test_dir
from tfrecord import TfRecord

# The global random seed.
tf.random.set_seed(tf_global_seed)
seednp(np_seed)
# The ratio of Validation set.
vald_ratio = 0.2
# The ratio of Test set.
test_ratio = 0.1

## Some examples to check the data augmentation: (for debugging).
#data_pth = "/your_desired_path/Data_Augm_Exper"
data_dir = Path(data_pth).with_suffix('')

# A dataset of all files in images and masks folders.
list_ds = tf.data.Dataset.list_files([str(data_dir/'*/*.png')], shuffle=False)

# Dataset size.
data_count = tf.data.experimental.cardinality(list_ds).numpy()//2
    
def normalize(input_img:tf.Tensor)->tf.Tensor:
    """Normalize the image color values to [0,1] range.

    Args:
        input_img (tf.Tensor): The input image of range [0,255].

    Returns:
        tf.Tensor: The processed image of range [0,1].
    """
    #input_image = tf.cast(input_img, tf.float32) / 255.0
    # MobileNet: covert to [-1,1] range
    input_image = -1 + tf.cast(input_img, tf.float32) / 127.5 
    return input_image


def load_image(input_img:str,input_mask:str)->tuple:
    """It converts the content of the string files in input_img & input_mask to 3D & 1D unit8 tensors.

    Args:
        input_img (str): The path of the input image of range [0,255].
        input_mask (str): The path of the input mask of values 0,1.

    Returns:
        input_img: The processed image of size (img_height,img_width) and values of range [0,1].
        input_mask: The processed mask of size (img_height,img_width).
    """
    # Convert the compressed string to a 3D uint8 tensor.
    input_img = tf.image.decode_image(input_img)    
    # Convert the compressed string to a 3D uint8 tensor.
    input_mask = tf.image.decode_image(input_mask,channels=1)
    #Is needed to make (tfio.experimental.color.rgba_to_rgb) works.
    r, g, b = input_img[:, :, 0], input_img[:, :, 1], input_img[:, :, 2]
    return tf.stack([r, g, b], axis=-1), input_mask


def process_path(file_path)-> tuple:
    """Process the given path of each image and mask.

    Args:
        file_path : the file path of the given image.

    Returns:
        tuple: Returns both the image and mask after being pre-processed.
    """
    file_path_img = tf.strings.regex_replace(file_path,"Masks_01","Processed")
    # Load the raw data from the file as a string.
    img = tf.io.read_file(file_path_img)
    # Load the raw data from the file as a string.
    mask = tf.io.read_file(file_path)
    img, mask = load_image(img,mask)   
    return img, mask


# Create the dataset.
img_ds = list_ds.take(data_count).shuffle(BUFFER_SIZE,seed=shuffle_data_seed).map(process_path, num_parallel_calls=AUTOTUNE)

# Split the dataset into Train, Valid, and Test sets.
test_size = int(data_count*test_ratio)
# Test dataset.
test_ds = img_ds.take(test_size)
# Save the test set.
makedirs(test_dir,exist_ok=True)
save_tf =TfRecord(test_ds,test_dir)
save_tf.generate_tf_record()
test_ds = None

# Validation dataset.
val_size = int(data_count*vald_ratio)
val_ds = img_ds.skip(test_size).take(val_size)
# Save the validation dataset
makedirs(val_dir,exist_ok=True)
tf_vald_obj = TfRecord(val_ds,val_dir)
tf_vald_obj.generate_tf_record()
# Train dataset.
train_size = data_count-val_size-test_size
train_ds_original = img_ds.skip(test_size+val_size).take(train_size)
#Save the original Training dataset
makedirs(train_dir,exist_ok=True)
tf_train_obj = TfRecord(train_ds_original,train_dir)
tf_train_obj.generate_tf_record()

# Create a `Counter` object and `Dataset.zip` it together with the training set.
counter = tf.data.Dataset.counter()
train_ds = tf.data.Dataset.zip((train_ds_original, (counter,counter)))
train_ds_original = None
val_ds = tf.data.Dataset.zip((val_ds, (counter,counter)))


@tf.py_function(Tout=tf.float32)
def random_rotate_image(image,rotation_angle):
    return ndimage_rotate(image,rotation_angle,reshape=False, mode='nearest')

def tf_random_rotate_image(image,rotation_angle):
    img_shape = image.shape
    image = random_rotate_image(image,rotation_angle)
    image.set_shape(img_shape)
    return image

# Augmentation layers #
### The first 5 cases, will be for augmenting image only.###
#0. Random image contrast.
class RandomContrast(tf.keras.Layer):
    def __init__(self,lower_value: float = 0.1,upper_value: float = 0.9):
        """Initialize Random Contrast Layer.

        Args:
            lower_value (float, optional): Lower bound for the random contrast factor. Defaults to 0.1.
            upper_value (float, optional): Upper bound for the random contrast factor. Defaults to 0.9.
        """
        super().__init__()
        self.lower = lower_value
        self.upper = upper_value
        
    def call(self, img_mask:Tuple, SEED:Tuple)->Tuple:
        """Change the contrast of the given image using the given seed.

        Args:
            img (tf.float32): An input image of size [IMG_SIZE,IMG_SIZE,3].
            mask (tf.unit8): An input mask of size [IMG_SIZE,IMG_SIZE,1].

        Returns:
            img, mask (tf.float32, tf.unit8): An augmented version of the given image and mask.
        """       
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]
    
        return tf.image.stateless_random_contrast(img,lower=self.lower,upper=self.upper,seed=new_seed),mask
 
 
#1. Adjust the hue of RGB image. 
class AdjustHue(tf.keras.Layer):
    def __init__(self,delta_value: float = 0.25):
        """Initialize Adjust the hue of RGB image Layer.

        Args:
            delta_value (float, optional): uses a delta_value randomly picked in the interval [-delta_value, delta_value). Defaults to 0.25.

        Raises:
            ValueError: delta_value shoule be in the ranage of [0, 0.5].
        """
        super().__init__()
        self.delta = delta_value
        if delta_value<0 or delta_value >0.5:
            raise ValueError("Delta value should be in the range [0,0.5], not %s" % delta_value)
        
    def call(self, img_mask:Tuple, SEED:Tuple)->Tuple:
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]
        
        return tf.image.stateless_random_hue(img,max_delta=self.delta,seed=new_seed),mask
    
    
#2. Radomize jpeg encoding quality.
class RandomizeJpeg(tf.keras.Layer):
    def __init__(self,min_quality: int = 0,max_quality:int =100):
        """Initialize Randomize jpeg encoding quality layer

        Args:
            min_quality (int, optional): Minimum jpeg encoding quality to use. Defaults to 0.
            max_quality (int, optional): Maximum jpeg encoding quality to use. Defaults to 100.

        Raises:
            ValueError: min_quality should be less than max_quality, and in the range [0,100]
        """
        super().__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality
        
        if min_quality<0 or max_quality<0 or min_quality>= max_quality:
            raise ValueError("min_qulaity and max_quality should be in the range [0,100] and min_quality<max_quality.")
        
    def call(self, img_mask:Tuple, SEED:Tuple)->Tuple:
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]
        
        return tf.image.stateless_random_jpeg_quality(img,
                                                      min_jpeg_quality=self.min_quality,
                                                      max_jpeg_quality=self.max_quality,
                                                      seed=new_seed),mask    


#3. Adjust the saturation of RGB images.
class ImageSaturation(tf.keras.Layer):
    def __init__(self,lower_sat_f: float = 0.1,upper_sat_f:float =5):
        """Initialize Adjust Image Saturation layer.

        Args:
            lower_sat_f (float, optional): Lower bound for the random saturation factor. Defaults to 0.1.
            upper_sat_f (float, optional): Upper bound for the random saturation factor. Defaults to 5.

        Raises:
            ValueError: lower_sat_f should be >=0, and upper_sat_f > lower_sat_f.
        """
        super().__init__()
        self.upper = upper_sat_f 
        self.lower = lower_sat_f
        
        if lower_sat_f<0 or upper_sat_f<=lower_sat_f:
            raise ValueError("Lower saturation factor should be greater than or equal to zero, and less than the Upper Saturation factor.")
        
    def call(self, img_mask:Tuple, SEED:Tuple)->Tuple:
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]
    
        return tf.image.stateless_random_saturation(img,lower=self.lower,upper=self.upper,seed=new_seed),mask
    

#4. Change the brightness of an image.
class ImageBrightness(tf.keras.Layer):
    def __init__(self,brightness_value: float = 0.95):
        """Initialize Adjust Brightness layer

        Args:
            brightness_value (float, optional): Adjust the brightness using the given value
                                                randomly picked in the interval [-brightness_value, 
                                                                                brightness_value).Defaults to 0.95.

        Raises:
            ValueError: In case brightness_value<0.
        """
        super().__init__()
        self.MAX_delta = brightness_value
        
        if brightness_value<0 :
            raise ValueError("The brightness factor should not be negative.")
        
    def call(self, img_mask:Tuple, SEED:Tuple)->Tuple:
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]
    
        return tf.image.stateless_random_brightness(img,max_delta=self.MAX_delta,
                                                                seed=new_seed),mask


### The remaining cases will augment Image & Mask.###        
#5. Random horizontal flip.
class HorizontalFlip(tf.keras.Layer):
    def __init__(self):
        """Initialize Horizontal Flipping layer.
        """
        super().__init__()
        
    def call(self, img_mask:Tuple, SEED:Tuple)->Tuple:
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]
    
        return tf.image.stateless_random_flip_left_right(img,new_seed),tf.image.stateless_random_flip_left_right(mask,new_seed)    
        

#6. Random vertical flip.
class VerticalFlip(tf.keras.Layer):
    def __init__(self):
        """Initialize Vertical Flipping layer.
        """
        super().__init__()
        
    def call(self, img_mask:Tuple, SEED:Tuple)->Tuple:
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]
    
        return tf.image.stateless_random_flip_up_down(img,new_seed),tf.image.stateless_random_flip_up_down(mask,new_seed)


#7. Random Rotation.
class RandomRotation(tf.keras.Layer):
    def __init__(self,rotation_angle:float=30):
        """Initialize Random Rotation layer.

        Args:
            den_value (float, optional): The rotation angle for rotating the image. Defaults to 30.
        """
        super().__init__()
        self.rotation_angle = rotation_angle
        
    def call(self, img_mask:Tuple, SEED:Tuple)->Tuple:
        img, mask = img_mask
            
        return tf_random_rotate_image(img,self.rotation_angle),tf.cast(tf_random_rotate_image(mask,self.rotation_angle),tf.uint8)


#8. Random Crop.
class RandomCrop(tf.keras.Layer):
    def __init__(self):
        """Initialize Random Crop layer.
        """
        super().__init__()
        
    def call(self, img_mask:Tuple, SEED:Tuple)->Tuple:
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]
        # Double the size of the image and mask to crop the desired size [IMG_SIZE,IMG_SIZE].
        img = tf.image.resize_with_crop_or_pad(img,IMG_SIZE*2,IMG_SIZE*2)
        mask = tf.image.resize_with_crop_or_pad(mask,IMG_SIZE*2,IMG_SIZE*2)
        return tf.image.stateless_random_crop(img,size=[IMG_SIZE,IMG_SIZE,3],seed=new_seed),tf.image.stateless_random_crop(mask, size=[IMG_SIZE,IMG_SIZE,1],seed=new_seed)
        

# Rotation by 90 degrees.
class Rotation90(tf.keras.Layer):
    def __init__(self,number_rotation:int=1):
        """Initialize 90 degrees rotation.
        """
        super().__init__()
        self.number_rotation = number_rotation
    
    def call(self,img_mask:Tuple, SEED:Tuple)->Tuple:
        img, mask = img_mask
        return tf.image.rot90(img,self.number_rotation),tf.image.rot90(mask,self.number_rotation)

# Augmentation class.
class DataAugmentation():
    def __init__(self,dataset:tf.data.Dataset,out_dir,tfrecord_obj:TfRecord):
        self.dataset = dataset
        self.out_dir = out_dir
        self.tfrecord = tfrecord_obj
        
    def augment_data(self):
        aug_method=[HorizontalFlip(),VerticalFlip(),RandomCrop(),Rotation90(),Rotation90(3)
                    ]  
              
        for idx,_aug_fun in enumerate(aug_method):
            new_aug_data = self.dataset.map(_aug_fun,num_parallel_calls=AUTOTUNE)
            self.tfrecord.dataset = new_aug_data
            self.tfrecord.generate_tf_record()
            new_aug_data = None
      

# Save the augmented Training dataset along with the original one.
aug_train_obj = DataAugmentation(train_ds,train_dir,tf_train_obj)
aug_train_obj.augment_data()
train_ds = None
# Save the augmented Validation dataset along with the original one.
aug_val_obj = DataAugmentation(val_ds,val_dir,tf_vald_obj)
aug_val_obj.augment_data()
val_ds = None