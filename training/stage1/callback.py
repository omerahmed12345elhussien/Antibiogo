"""
* callback.py contains callsbacks that are needed for saving the models and predictions during training.

"""
import tensorflow as tf
from tensorflow.keras.utils import array_to_img
from dataloader import single_batch
import wandb
from utils import create_mask


for img,mask in single_batch:
    sample_image, sample_mask=img[0],mask[0]
# Callbacks 
EarlyStopping_callback = tf.keras.callbacks.EarlyStopping(patience=3)  
checkpoint_filepath = 'BaseModel/checkpoint.model.keras'
savemodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        mode='min'
                                                        )

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        wandb.log({"Prediction": [wandb.Image(array_to_img(sample_image), caption="Input Image"),
                                  wandb.Image(array_to_img(sample_mask[...,tf.newaxis]), caption="True Mask"),
                                  wandb.Image(array_to_img(create_mask(self.model.predict(sample_image[tf.newaxis, ...]))),
                                              caption="Predicted Mask start_of_training")]})
    def on_train_end(self, logs=None):
        wandb.log({"Prediction": [wandb.Image(array_to_img(sample_image), caption="Input Image"),
                                  wandb.Image(array_to_img(sample_mask[...,tf.newaxis]), caption="True Mask"),
                                  wandb.Image(array_to_img(create_mask(self.model.predict(sample_image[tf.newaxis, ...]))),
                                              caption="Predicted Mask end_of_training")]})    
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch%10==0:
            wandb.log({"Prediction": [wandb.Image(array_to_img(sample_image), caption="Input Image"),
                                      wandb.Image(array_to_img(sample_mask[...,tf.newaxis]), caption="True Mask"),
                                      wandb.Image(array_to_img(create_mask(self.model.predict(sample_image[tf.newaxis, ...]))),
                                                  caption="Predicted Mask epoch:{}".format(epoch))]})


class CustomWandbMetricsLogger(tf.keras.callbacks.Callback):
    """ Logger that sends system metrics to W&B."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before WandbMetricsLogger()"
            )
            
    def on_epoch_end(self, epoch, logs=None):
        logs = dict() if logs is None else {f"epoch/{k}": v for k, v in logs.items()}
        logs["epoch/epoch"] = epoch
        wandb.log(logs)
       