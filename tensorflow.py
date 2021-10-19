import pandas as pd
import tensorflow as tf
from datetime import datetime


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        print("End epoch {} of training; got log keys: {}".format(epoch, logs))
        # commento perchè non splitto il dataset in train-val
        #tf.summary.scalar('validation/val_loss', data=logs['val_loss'], step=epoch)
        tf.summary.scalar('validation/mse', data=logs['loss'], step=epoch)
        #tf.summary.flush()

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))

X_true_tf = tf.convert_to_tensor(X_true)
y_tf = tf.convert_to_tensor(y)

# The kernel, through kernel_regularizer, which applies regularization to the kernel a.k.a. the actual weights;
# The bias value, through bias_regularizer, which applies regularization to the bias, which shifts the layer outputs;
# The activity value, through activity_regularizer, which applies the regularizer to the output of the layer, i.e. the activation value (which is the combination of the weights + biases with the input vector, fed through the activation function) (Tonutti, 2017).
input = tf.keras.Input(shape=(X_true_tf.shape[1],),dtype='float64',name='input')   # shape(1 riga,4 feature)
act = tf.keras.layers.Activation(activation='linear')(input)
output = tf.keras.layers.Dense(units=1, name='output')(act)
model = tf.keras.Model(inputs=input,outputs=output)
model.summary()
tf.keras.utils.plot_model(model, to_file="my_first_model_with_shape_info.png", show_shapes=True)

log_dir = "/home/davide/PycharmProjects/dnn_linearReg/tensorboard-logs"+ \
          datetime.now().strftime("%Y%m%d-%H%M%S")

file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

"""
mkdir tensorboard-logs
{tensorboard --logdir <"log_dir path">}  
tf.summary.scalar
tf.summary.image
tf.summary.text
tf.summary.histogram
tf.summary.create_file_writer
tf.summary.write
"""

model.compile(optimizer=tf.optimizers.SGD(learning_rate=1e-2),
              loss='mean_squared_error',
              metrics=['mean_squared_error'])
modellofittato= model.fit(X_true_tf,y_tf,
          epochs=200,
          validation_split=0.0,
          verbose=0,
          callbacks=[tensorboard_callback, CustomCallback()]) #tolto lr_schedule da callbacks
# history =pd.DataFrame(modellofittato.history)

# get weights ti ritorna due liste: la prima sono i pesi del layer (solo 1 qui)
# e il secondo sono i termini bias
pesi=pd.DataFrame([model.get_weights()[0].ravel()],columns=X_true.columns)
biases= pd.DataFrame(model.get_weights()[1],columns=['bias'])
df_pesi_tf=pd.concat([pesi,biases],axis=1)
