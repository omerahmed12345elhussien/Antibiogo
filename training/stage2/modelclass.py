"""
* modelclass.py contains The modified training and testing steps of tf.keras.Model class.
* we are only interested on specific metrics, and we focused mainly on them.

"""
import tensorflow as tf
from utils_copy import OUTPUT_CLASSES

class CustomModel(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss_tracker = tf.keras.metrics.Mean(name="loss",dtype=tf.float64)
    self.meaniou_metric = tf.keras.metrics.MeanIoU(num_classes=OUTPUT_CLASSES,name="iou",dtype=tf.float64)
    self.acc_metric = tf.keras.metrics.Accuracy(name="accuracy",dtype=tf.float64)

  def train_step(self, data):
    # Unpack the data.
    image, mask = data
    # Open a GradientTape.
    with tf.GradientTape() as tape:
      # Forward pass.
      logits = self(image,training=True)
      # Compute the loss.
      loss_value = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true=mask, y_pred=logits)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    trainable_vars = self.trainable_variables
    grads = tape.gradient(loss_value, trainable_vars)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    self.optimizer.apply_gradients(zip(grads, trainable_vars))
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    self.meaniou_metric.update_state(mask, tf.math.argmax(logits, axis=-1))
    self.acc_metric.update_state(mask, tf.math.argmax(logits, axis=-1))
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "iou": self.meaniou_metric.result(),"accuracy":self.acc_metric.result()}
  
  def test_step(self, data):
    # Unpack the data.
    image, mask = data
    # Compute predictions
    pred_mask = self(image, training=False)
    # Compute the loss.
    loss_value = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true=mask, y_pred=pred_mask)
    # Update the metrics.
    self.loss_tracker.update_state(loss_value)
    self.meaniou_metric.update_state(mask, tf.math.argmax(pred_mask, axis=-1))
    self.acc_metric.update_state(mask, tf.math.argmax(pred_mask, axis=-1))
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "iou": self.meaniou_metric.result(),"accuracy":self.acc_metric.result()}

  @property
  def metrics(self):
    return [self.loss_tracker, self.meaniou_metric,self.acc_metric]

  
class TrainingLoop():
  def __init__(self, model:tf.keras.Model, criterion, train_data_loader,val_data_loader,
               optimizer, callbacks_list:list, num_epochs:int,train_decay:float=0.9999,
               val_decay:float=0.9999)->None:
    self.model = model
    self.criterion = criterion
    self.train_data_loader = train_data_loader
    self.val_data_loader = val_data_loader
    self.optimizer = optimizer
    self.num_epochs = num_epochs
    self.train_decay=train_decay
    self.val_decay = val_decay
    self.acc_metric = tf.keras.metrics.Accuracy(name="accuracy",dtype=tf.float64)
    self.meaniou_metric = tf.keras.metrics.MeanIoU(num_classes=OUTPUT_CLASSES,name="iou",dtype=tf.float64,ignore_class=0)
    self.callbacks = tf.keras.callbacks.CallbackList(callbacks_list,
                                                     add_history=True,
                                                     model=model
                                                     )
  
  @tf.function
  def _train_step(self,image, mask, weight_mat):
    # Open a GradientTape.
    with tf.GradientTape() as tape:
      # Forward pass.
      logits = self.model(image,training=True)
      # Compute the loss.
      loss_value = self.criterion(y_true=mask, y_pred=logits, sample_weight=weight_mat)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    trainable_vars = self.model.trainable_weights
    grads = tape.gradient(loss_value, trainable_vars)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    self.optimizer.apply_gradients(zip(grads, trainable_vars))
    self.meaniou_metric.update_state(mask, tf.math.argmax(logits, axis=-1))
    miou_value = self.meaniou_metric.result()
    self.meaniou_metric.reset_state()
    self.acc_metric.update_state(mask, tf.math.argmax(logits, axis=-1))
    acc_value = self.acc_metric.result()
    self.acc_metric.reset_state()
    return loss_value, miou_value,acc_value
  
  @tf.function
  def _test_step(self,image, mask, weight_mat):
    # Compute predictions
    pred_mask = self.model(image, training=False)
    # Compute the loss.
    loss_value = self.criterion(y_true=mask, y_pred=pred_mask, sample_weight=weight_mat)
    self.meaniou_metric.update_state(mask, tf.math.argmax(pred_mask, axis=-1))
    miou_value = self.meaniou_metric.result()
    self.meaniou_metric.reset_state()
    self.acc_metric.update_state(mask, tf.math.argmax(pred_mask, axis=-1))
    acc_value = self.acc_metric.result()
    self.acc_metric.reset_state()
    return loss_value, miou_value, acc_value
  
  def train(self):
    """Simple training loop for a tensorflow model.
    """
    # Exponential moving average of the: loss, accuracy, mIoU.
    self.ema_loss = None
    self.ema_loss_val = None
    self.ema_acc = None
    self.ema_acc_val = None
    self.ema_miou = None
    self.ema_miou_val = None
    self.model.stop_training = False
    self.callbacks.on_train_begin()
    logs = {}
    # Loop over epochs.
    for epoch in range(self.num_epochs):
      # Iterate over the batches of the dataset.
      for step, (x_batch_train, y_batch_train, weight_batch_train) in enumerate(self.train_data_loader):
        loss_value, miou_value, acc_value = self._train_step(x_batch_train,y_batch_train,weight_batch_train)
        if (self.ema_loss is None) and (self.ema_miou is None) and (self.ema_acc is None):
          self.ema_loss = loss_value
          self.ema_acc = acc_value
          self.ema_miou = miou_value
        else:
          self.ema_loss -= (1-self.train_decay)*(self.ema_loss-loss_value)
          self.ema_acc -= (1-self.train_decay)*(self.ema_acc-acc_value)
          self.ema_miou -= (1-self.train_decay)*(self.ema_miou-miou_value)
        miou_value,loss_value,acc_value=None,None,None
        #self.callbacks.on_train_batch_end(step, logs)
        if self.model.stop_training: break
      # Run a validation loop at the end of each epoch.
      for step, (x_batch_val, y_batch_val, weight_batch_val) in enumerate(self.val_data_loader):
        loss_value, miou_value_val, acc_value_val = self._test_step(x_batch_val,y_batch_val,weight_batch_val)
        if (self.ema_loss_val is None) and (self.ema_miou_val is None) and (self.ema_acc_val is None):
          self.ema_loss_val = loss_value
          self.ema_acc_val = acc_value_val
          self.ema_miou_val = miou_value_val
        else:
          self.ema_loss_val -= (1-self.val_decay)*(self.ema_loss_val-tf.identity(loss_value))
          self.ema_acc_val -= (1-self.val_decay)*(self.ema_acc_val-acc_value_val)
          self.ema_miou_val -= (1-self.val_decay)*(self.ema_miou_val-miou_value_val)
        loss_value,miou_value_val,acc_value_val=None,None,None
      # Log metrics at the end of each epoch.
      logs = {'loss':self.ema_loss,'iou':tf.keras.ops.round(self.ema_miou,4),
              'learning_rate':tf.cast(self.optimizer.learning_rate,tf.float32),
              'val_loss':self.ema_loss_val,
              'val_iou':tf.keras.ops.round(self.ema_miou_val,4),
              'accuracy':tf.keras.ops.round(self.ema_acc,4),
              'val_accuracy':tf.keras.ops.round(self.ema_acc_val,4)
              }
      self.callbacks.on_epoch_end(epoch,logs)
      if self.model.stop_training: break
    self.callbacks.on_train_end(logs)


class FocalLossTraining():
  def __init__(self, model:tf.keras.Model, criterion, train_data_loader,val_data_loader,
               optimizer, callbacks_list:list, num_epochs:int,train_decay:float=0.9999,
               val_decay:float=0.9999)->None:
    self.model = model
    self.criterion = criterion
    self.train_data_loader = train_data_loader
    self.val_data_loader = val_data_loader
    self.optimizer = optimizer
    self.num_epochs = num_epochs
    self.train_decay=train_decay
    self.val_decay = val_decay
    self.acc_metric = tf.keras.metrics.Accuracy(name="accuracy",dtype=tf.float64)
    self.meaniou_metric = tf.keras.metrics.MeanIoU(num_classes=OUTPUT_CLASSES,name="iou",dtype=tf.float64,ignore_class=0)
    self.callbacks = tf.keras.callbacks.CallbackList(callbacks_list,
                                                     add_history=True,
                                                     model=model
                                                     )
  
  @tf.function
  def _train_step(self,image, mask):
    # Open a GradientTape.
    with tf.GradientTape() as tape:
      # Forward pass.
      logits = self.model(image,training=True)
      # Compute the loss.
      loss_value = self.criterion(y_true=tf.one_hot(mask,OUTPUT_CLASSES), y_pred=logits)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    trainable_vars = self.model.trainable_weights
    grads = tape.gradient(loss_value, trainable_vars)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    self.optimizer.apply_gradients(zip(grads, trainable_vars))
    self.meaniou_metric.update_state(mask, tf.math.argmax(logits, axis=-1))
    miou_value = self.meaniou_metric.result()
    self.meaniou_metric.reset_state()
    self.acc_metric.update_state(mask, tf.math.argmax(logits, axis=-1))
    acc_value = self.acc_metric.result()
    self.acc_metric.reset_state()
    return loss_value, miou_value,acc_value
  
  @tf.function
  def _test_step(self,image, mask):
    # Compute predictions
    pred_mask = self.model(image, training=False)
    # Compute the loss.
    loss_value = self.criterion(y_true=tf.one_hot(mask,OUTPUT_CLASSES), y_pred=pred_mask)
    self.meaniou_metric.update_state(mask, tf.math.argmax(pred_mask, axis=-1))
    miou_value = self.meaniou_metric.result()
    self.meaniou_metric.reset_state()
    self.acc_metric.update_state(mask, tf.math.argmax(pred_mask, axis=-1))
    acc_value = self.acc_metric.result()
    self.acc_metric.reset_state()
    return loss_value, miou_value, acc_value
  
  def train(self):
    """Simple training loop for a tensorflow model.
    """
    # Exponential moving average of the: loss, accuracy, mIoU.
    self.ema_loss = None
    self.ema_loss_val = None
    self.ema_acc = None
    self.ema_acc_val = None
    self.ema_miou = None
    self.ema_miou_val = None
    self.model.stop_training = False
    self.callbacks.on_train_begin()
    logs = {}
    # Loop over epochs.
    for epoch in range(self.num_epochs):
      # Iterate over the batches of the dataset.
      for step, (x_batch_train, y_batch_train) in enumerate(self.train_data_loader):
        loss_value, miou_value, acc_value = self._train_step(x_batch_train,y_batch_train)
        if (self.ema_loss is None) and (self.ema_miou is None) and (self.ema_acc is None):
          self.ema_loss = loss_value
          self.ema_acc = acc_value
          self.ema_miou = miou_value
        else:
          self.ema_loss -= (1-self.train_decay)*(self.ema_loss-loss_value)
          self.ema_acc -= (1-self.train_decay)*(self.ema_acc-acc_value)
          self.ema_miou -= (1-self.train_decay)*(self.ema_miou-miou_value)
        miou_value,loss_value,acc_value=None,None,None
        if self.model.stop_training: break
      # Run a validation loop at the end of each epoch.
      for step, (x_batch_val, y_batch_val) in enumerate(self.val_data_loader):
        loss_value, miou_value_val, acc_value_val = self._test_step(x_batch_val,y_batch_val)
        if (self.ema_loss_val is None) and (self.ema_miou_val is None) and (self.ema_acc_val is None):
          self.ema_loss_val = loss_value
          self.ema_acc_val = acc_value_val
          self.ema_miou_val = miou_value_val
        else:
          self.ema_loss_val -= (1-self.val_decay)*(self.ema_loss_val-tf.identity(loss_value))
          self.ema_acc_val -= (1-self.val_decay)*(self.ema_acc_val-acc_value_val)
          self.ema_miou_val -= (1-self.val_decay)*(self.ema_miou_val-miou_value_val)
        loss_value,miou_value_val,acc_value_val=None,None,None
      # Log metrics at the end of each epoch.
      logs = {'loss':self.ema_loss,'iou':tf.keras.ops.round(self.ema_miou,4),
              'learning_rate':tf.cast(self.optimizer.learning_rate,tf.float32),
              'val_loss':self.ema_loss_val,
              'val_iou':tf.keras.ops.round(self.ema_miou_val,4),
              'accuracy':tf.keras.ops.round(self.ema_acc,4),
              'val_accuracy':tf.keras.ops.round(self.ema_acc_val,4)
              }
      self.callbacks.on_epoch_end(epoch,logs)
      if self.model.stop_training: break
    self.callbacks.on_train_end(logs)
    
  def evaluate(self,test_batch,test_decay:float=0.9999):
    # Exponential moving average of the: loss, accuracy, mIoU.
    self.ema_loss_val = None
    self.ema_acc_val = None
    self.ema_miou_val = None
    # Run a validation loop at the end of each epoch.
    for step, (x_batch_val, y_batch_val) in enumerate(test_batch):
      loss_value, miou_value_val, acc_value_val = self._test_step(x_batch_val,y_batch_val)
      if (self.ema_loss_val is None) and (self.ema_miou_val is None) and (self.ema_acc_val is None):
        self.ema_loss_val = loss_value
        self.ema_acc_val = acc_value_val
        self.ema_miou_val = miou_value_val
      else:
        self.ema_loss_val -= (1-test_decay)*(self.ema_loss_val-tf.identity(loss_value))
        self.ema_acc_val -= (1-test_decay)*(self.ema_acc_val-acc_value_val)
        self.ema_miou_val -= (1-test_decay)*(self.ema_miou_val-miou_value_val)
      loss_value,miou_value_val,acc_value_val=None,None,None
    return {"loss": self.ema_loss_val, "iou": tf.keras.ops.round(self.ema_miou_val,4),"accuracy":tf.keras.ops.round(self.ema_acc_val,4)}

