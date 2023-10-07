import keras
from keras import backend as K
from keras.layers import Activation, Dense, Input, Subtract, BatchNormalization
from keras import initializers, regularizers
from keras.models import Model
import numpy as np
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import ndcg_score
import random

np.random.seed(1)
random.seed(1)


def cos_sin(x):
    return tf.concat([tf.math.cos(x), tf.math.sin(x)], axis=-1)


class NLinear(keras.layers.Layer):
    def __init__(self, n, d_in, d_out):
        super(NLinear, self).__init__()
        self.weight = self.add_weight(shape=(n, d_in, d_out), initializer="glorot_uniform",
                                      trainable=True, name='plr_weight')
        self.bias = self.add_weight(shape=(n, d_out), initializer="zeros",
                                    trainable=True, name='plr_bias')

    def call(self, x):
        x = tf.expand_dims(x, axis=-1) * self.weight
        x = tf.reduce_sum(x, axis=-2)
        x = x + self.bias
        return x


class PeriodicEmbedding(keras.layers.Layer):
    def __init__(self, n_features, n, sigma):
        super(PeriodicEmbedding, self).__init__()
        coefficients = tf.random.normal(shape=(n_features, n), mean=0.0, stddev=sigma)
        self.coefficients = self.add_weight(shape=(n_features, n),
                                            initializer=tf.keras.initializers.Constant(coefficients),
                                            trainable=True,
                                            name='plr_coefficients')

    def call(self, x):
        assert x.shape.ndims == 2
        return cos_sin(2 * tf.constant(np.pi) * tf.expand_dims(self.coefficients, axis=0) * tf.expand_dims(x, axis=-1))


class NumEmbeddings(keras.models.Model):
    def __init__(self, n_features, d_embedding, n=10, sigma=0.02):
        super(NumEmbeddings, self).__init__()
        self.model_layers = keras.models.Sequential([
            PeriodicEmbedding(n_features, n, sigma),
            NLinear(n_features, n * 2, d_embedding),
            keras.layers.ReLU()
        ])

    def call(self, x):
        return self.model_layers(x)


class RankerNN(object):

    def __init__(self):
        """
        Parameters
        ----------
        input_size : integer
            Number of input features.
        hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
            The ith element represents the number of neurons in the ith
            hidden layer.
        activation : tuple, length = n_layers - 2, default ('relu',)
            The ith element represents activation function in the ith
            hidden layer.
        solver : {'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', adamax},
        default 'adam'
            The solver for weight optimization.
            - 'adam' refers to a stochastic gradient-based optimizer proposed
              by Kingma, Diederik, and Jimmy Ba
        """

    def build_model(self, input_shape, weight_decay, lr, solver, initializer=None,
                    d_embedding=10, n=10, sigma=0.02, hidden=16):
        """
        Build Keras Ranker NN model (Ranknet / LambdaRank NN).
        """
        # Neural network structure
        num_embeddings = NumEmbeddings(input_shape, d_embedding, n, sigma)
        hidden_layers = []
        if initializer is not None:
            my_init = initializer
        else:
            my_init = initializers.he_normal(seed=0)
        hidden_layers.append(Dense(hidden, activation='relu', name=str('relu') + '_layer' + str(0),
                                   kernel_initializer=my_init, kernel_regularizer=regularizers.l2(weight_decay),
                                   bias_regularizer=regularizers.l2(weight_decay)))
        hidden_layers.append(BatchNormalization())
        hidden_layers.append(Dense(hidden // 4, activation='relu', name=str('relu') + '_layer' + str(1),
                                   kernel_initializer=my_init, kernel_regularizer=regularizers.l2(weight_decay),
                                   bias_regularizer=regularizers.l2(weight_decay)))
        h0 = Dense(1, activation='linear', name='Identity_layer')
        input1 = Input(shape=(input_shape,), name='Input_layer1')
        input2 = Input(shape=(input_shape,), name='Input_layer2')
        x1 = num_embeddings.call(input1)
        x2 = num_embeddings.call(input2)
        flatten_layer = tf.keras.layers.Flatten()
        x1 = flatten_layer(x1)
        x2 = flatten_layer(x2)
        for i in range(len(hidden_layers)):
            x1 = hidden_layers[i](x1)
            x2 = hidden_layers[i](x2)
        x1 = h0(x1)
        x2 = h0(x2)
        # Subtract layer
        subtracted = Subtract(name='Subtract_layer')([x1, x2])
        # sigmoid
        out = Activation('sigmoid', name='Activation_layer')(subtracted)
        # build model
        model = Model(inputs=[input1, input2], outputs=out)
        if solver == 'adam':
            adam = keras.optimizers.Adam(lr=lr)
            model.compile(optimizer=adam, loss="binary_crossentropy")
        else:
            model.compile(optimizer=solver, loss="binary_crossentropy")
        self.model = model
        return model

    def fit(self, X1_trans, X2_trans, y_trans, weight, batch_size=None, epochs=1, verbose=1,
            validation_split=0.0, patience=10, validation_data=None,
            val_data_for_ndcg=None):
        """Transform data and fit model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        y : array, shape (n_samples,)
            Target labels.
        qid: array, shape (n_samples,)
            Query id that represents the grouping of samples.
        """
        val_data, val_label = val_data_for_ndcg
        features = ['f%d' % i for i in range(11)]

        class CustomEarlyStopping(keras.callbacks.Callback, RankerNN):
            def __init__(self, val_data, val_label, patience):
                super(CustomEarlyStopping, self).__init__()
                self.val_data = val_data
                self.val_label = val_label
                self.patience = patience
                self.count = 0
                self.best_val_loss = float('-inf')

            def on_epoch_end(self, epoch, logs):
                ndcg_list = []
                for file in self.val_data['filename'].unique():
                    data_tmp = self.val_data[self.val_data['filename'] == file][features]
                    label_tmp = self.val_label.loc[data_tmp.index]
                    ndcg_list.append(self.evaluate(data_tmp.values, label_tmp.values.ravel()))
                val_loss = np.mean(ndcg_list)
                print()
                print(f"NDCG at epoch {epoch} is {val_loss}")
                if val_loss > self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.count = 0
                    print("Best epoch!")
                else:
                    self.count += 1
                    if self.count >= patience:
                        self.model.stop_training = True
                        print("Early stopping triggered at epoch ", epoch + 1)

        my_early_stopping = CustomEarlyStopping(val_data, val_label, patience)
        history = self.model.fit([X1_trans, X2_trans], y_trans, sample_weight=weight, batch_size=batch_size,
                                 epochs=epochs, callbacks=[my_early_stopping],
                                 verbose=verbose)
        return my_early_stopping.best_val_loss

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)

    def finetune(self, X1_trans, X2_trans, y_trans, weight, batch_size, lr,
                 epochs=1, verbose=1, patience=10, validation_data=None):
        keras.backend.set_value(self.model.optimizer.lr, lr)
        print("The learning rate now is %lf." % keras.backend.get_value(self.model.optimizer.lr))
        early_stopping_monitor = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            restore_best_weights=True
        )
        self.model.fit([X1_trans, X2_trans], y_trans, sample_weight=weight, batch_size=batch_size, epochs=epochs,
                       callbacks=[early_stopping_monitor], verbose=verbose,
                       validation_data=validation_data)

    def evaluate(self, data, label):
        pred = self.predict(data)
        ndcg = ndcg_score(np.array([label]), np.array([pred]))
        return ndcg

    def predict(self, X):
        """Predict output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        Returns
        -------
        y_pred: array, shape (n_samples,)
            Model prediction.
        """
        ranker_output = K.function([self.model.layers[0].input], [self.model.layers[-3].get_output_at(0)])
        return ranker_output([X])[0].ravel()


class RankNetNN(RankerNN):

    def __init__(self):
        super(RankNetNN, self).__init__()


class LambdaRankNN(RankerNN):

    def __init__(self):
        super(LambdaRankNN, self).__init__()
