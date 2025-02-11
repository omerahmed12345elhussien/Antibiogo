"""
    * train.py contains the script to train our models and reporting the results to weight and biases.
"""
import tensorflow as tf
from dataloader import train_batches, vald_batches
from basemodel import unet_model
from callback import savemodel_callback,DisplayCallback,EarlyStopping_callback,CustomWandbMetricsLogger
from utils import OUTPUT_CLASSES, optimAdam
from modelclass import TrainingLoop
import wandb
from os import getenv

if __name__=="__main__":
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    EPOCHS = 100
    #model_net_MobileNetv2
    model = unet_model(output_channels=OUTPUT_CLASSES)
    
    wandb.login(key="add_your_api_key")
    # Launch an experiment
    wandb.init(
        project="MNetv2_256", #Project name.
        tags=["MNetv2_v35_C512-C256-C128-C64"],
        config={
            "epoch": EPOCHS,
            "batch_size":int(getenv("BATCH_SIZE")),
            "weight_decay":float(getenv("Weight_Decay")),
            "Trial_number":int(getenv("TRIAL_NUMBER")),
            "Epoch_Interval":int(getenv("EPOCH_INTERV")),
            "Alpha":float(getenv("ALPHA"))
        },
    )
    config = wandb.config
    # Add WandbMetricsLogger to log metrics 
    wandb_callbacks =CustomWandbMetricsLogger()
    callback_option=[wandb_callbacks,DisplayCallback(),savemodel_callback,EarlyStopping_callback]
    
    train_obj = TrainingLoop(model,criterion,train_batches,vald_batches,optimAdam,callback_option,EPOCHS,0.99999)
    train_obj.train()
    # Mark the run as finished
    wandb.finish()
    
    