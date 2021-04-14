import kerastuner as kt
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def build_model(hp):
    """Builds a convolutional model."""
    inputs = tf.keras.Input(shape=(1024, 3))
    x = inputs

    # Activation function
    activation_choice = hp.Choice('Activation', ['leaky_relu', 'relu', 'elu'])
    if activation_choice == 'relu':
        activation = tf.keras.layers.ReLU()
    elif activation_choice == 'leaky_relu':
        activation = tf.keras.layers.LeakyReLU(alpha=0.3)
    elif activation_choice == 'elu':
        activation = tf.keras.layers.ELU()

    # Batch normalization
    apply_batch_norm = hp.Choice('Batch Norm', ['do_batchnorm', 'not_batchnorm'])

    # Conv layers
    for i in range(hp.Int('conv_layers', 1, 6, default=6)):
        x = tf.keras.layers.Conv1D(
            filters=hp.Int('filters_' + str(i), 4, 96, step=4, default=96),
            kernel_size=hp.Int('kernel_size_' + str(i), 3, 65),
            activation='linear',
            padding='same')(x)

        if hp.Choice('pooling' + str(i), ['max', 'avg']) == 'max':
            x = tf.keras.layers.MaxPooling1D()(x)
        else:
            x = tf.keras.layers.AveragePooling1D()(x)

        if apply_batch_norm == 'do_batchnorm':
            x = tf.keras.layers.BatchNormalization()(x)
        x = activation(x)

    global_effect = hp.Choice('global_pooling', ['max', 'avg', 'flat'])
    if global_effect == 'max':
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
    elif global_effect == 'flat':
        x = tf.keras.layers.Flatten()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense Layers
    dropout = hp.Choice('Dropout', ['OG_dropout', 'Gaussian', 'Alpha'])
    for i in range(hp.Int('dense_layers', 1, 3, default=3)):
        x = tf.keras.layers.Dense(hp.Int('dense_neurons_' + str(i), 3, 4096),
            activation='linear'
            )(x)

        if dropout == 'OG_Dropout':
            x = tf.keras.layers.Dropout( hp.Float('drop_' + str(i), .0, .5) )(x)
        elif dropout == 'Gaussian':
            x = tf.keras.layers.GaussianDropout( hp.Float('drop_' + str(i), .0, .5) )(x)
        elif dropout == 'Alpha':
            x = tf.keras.layers.AlphaDropout( hp.Float('drop_' + str(i), .0, .5) )(x)
            
    outputs = tf.keras.layers.Dense(15, activation='linear')(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer = 'adam' #hp.Choice('optimizer', ['adam', 'sgd'])
    model.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])
    return model


class MyTuner(kt.Tuner):

    def run_trial(self, trial, train_ds):
        hp = trial.hyperparameters

        # Hyperparameters can be added anywhere inside `run_trial`.
        # When the first trial is run, they will take on their default values.
        # Afterwards, they will be tuned by the `Oracle`.
        train_ds = train_ds.batch(
            hp.Int('batch_size', 32, 512, step=32, default=64))

        model = self.hypermodel.build(trial.hyperparameters)
        lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
        optimizer = tf.keras.optimizers.Adam(lr)
        epoch_loss_metric = tf.keras.metrics.Mean()

        @tf.function
        def run_train_step(data):
            images = tf.dtypes.cast(data[0]['image'], 'float32') / 36.
            labels = data[1]['label']
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = tf.keras.losses.MSE(
                    labels, logits)
                # Add any regularization losses.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_metric.update_state(loss)
            return loss

        # `self.on_epoch_end` reports results to the `Oracle` and saves the
        # current state of the Model. The other hooks called here only log values
        # for display but can also be overridden. For use cases where there is no
        # natural concept of epoch, you do not have to call any of these hooks. In
        # this case you should instead call `self.oracle.update_trial` and
        # `self.oracle.save_model` manually.
        for epoch in range(10):
            print('Epoch: {}'.format(epoch))

            self.on_epoch_begin(trial, model, epoch, logs={})
            for batch, data in enumerate(train_ds):
                self.on_batch_begin(trial, model, batch, logs={})
                batch_loss = run_train_step(data)
                self.on_batch_end(trial, model, batch, logs={'loss': batch_loss})

                if batch % 100 == 0:
                    loss = epoch_loss_metric.result().numpy()
                    print('Batch: {}, Average Loss: {}'.format(batch, loss))

            epoch_loss = epoch_loss_metric.result().numpy()
            self.on_epoch_end(trial, model, epoch, logs={'loss': epoch_loss})
            epoch_loss_metric.reset_states()


def main(train_dataset, val_dataset):
  tuner = MyTuner(
      oracle=kt.oracles.BayesianOptimization(
          objective=kt.Objective('loss', 'min'),
          max_trials=20),
      hypermodel=build_model,
      directory='results',
      project_name='run_12_hyperparam_tuning')

  # Load mnist data
#  mnist_data = tfds.load('mnist')
#  mnist_train, mnist_test = mnist_data['train'], mnist_data['test']
#  mnist_train = mnist_train.shuffle(1000)
#  print(mnist_train)
#  exit()
  
  # Perform hyperparameter search
  tuner.search(train_ds=train_dataset)

  best_hps = tuner.get_best_hyperparameters()[0]
  print(best_hps.values)

  best_model = tuner.get_best_models()[0]

if __name__ == '__main__':
  main()

