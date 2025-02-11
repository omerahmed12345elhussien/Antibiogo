"""
* train.py contains the training script for models that crop the petri dish and identify the antibiotic disks using focal loss.
* this script is recommended for cluster use only (since several hyperparameters are needed to be passed to have it working).
"""
import tensorflow as tf
from tfrecord_loader import train_batches, vald_batches
from basemodel import unet_model
from utils_copy import optimAdam
from os import getenv,path
from modelclass import FocalLossTraining
from tfrecord_callback import savemodel_callback,DisplayCallback,EarlyStopping_callback,CustomWandbMetricsLogger
import wandb

if __name__=="__main__":
    criterion = tf.keras.losses.CategoricalFocalCrossentropy(from_logits=True)
    EPOCHS = int(getenv("EPOCHS"))#100
    if int(getenv("MODEL_OPTION"))==1:
        model = unet_model()#model_net_MobileNetv2
    elif int(getenv("MODEL_OPTION"))==3:
        model = tf.keras.models.load_model(path.join(getenv("ROOT_DIR_ST2"),"Training/BaseModel_st2",getenv("MODEL_NAME")))
    elif int(getenv("MODEL_OPTION"))==4:
      model = unet_model(model_version=3)#model_net_MobileNetv3
    elif int(getenv("MODEL_OPTION"))==5:
      model = unet_model(model_version=4)
    wandb.login(key="add_your_api_key")
    # Launch an experiment
    wandb.init(
        project="Tfrecord_512_st2st1",
        tags=[getenv("TAGS")],#"MNetv2_v35_C64-C32-C16-C8"],
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
    if int(getenv("CALLBACK"))==1:
        callback_option=[wandb_callbacks,DisplayCallback(),savemodel_callback,EarlyStopping_callback]
    elif int(getenv("CALLBACK"))==2:
        callback_option=[wandb_callbacks,DisplayCallback(),savemodel_callback]
    elif int(getenv("CALLBACK"))==3:
        callback_option=[wandb_callbacks,DisplayCallback(),EarlyStopping_callback]
    else:
        callback_option=[wandb_callbacks,DisplayCallback()]
    
    train_obj = FocalLossTraining(model,criterion,train_batches,vald_batches,optimAdam,callback_option,EPOCHS,0.99999)
    train_obj.train()
    # Mark the run as finished
    wandb.finish()
    