import os
from xmlrpc.client import Boolean

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
import sys
import random

# sys.path.append("../utile")
from database import create_cnx, config_parse
import pandas as pd
import numpy as np
import os
from typing import List


path = "../data"
import base64
import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Embedding,
    Concatenate,
    Flatten,
    Dense,
    Dropout,
    concatenate,
)
from sklearn.cluster import KMeans

from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import InputSpec  # , Normalization, BatchNormalization
import keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import SGD, Adam


# set random seeds
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
######################## Regression based embedding ########################


class Deep_Embedding_Model(object):
    """
    The aim of this class is to provide a support for dense model to train embeddings
    """

    def __init__(
        self,
        l_shape: List[int] = [100, 50, 25],
        dropout: List[float] = [0.2, 0.2],
        activation_functions: List[str] = ["relu", "relu", "relu"],
        k_embedding: dict = None,
        dict_shape: dict = {},
        outputs_shape: int = None,
        model_name: str = "Deep_embedded_model",
        numeric_features: List[str] = None,
        textual_features: List[str] = None,
        ohe_features: List[str] = None,
        image: Boolean = False,
        image_shape: List[int] = [4, 4, 64],
        image_layers_shape: List[int] = [512, 256, 128],
    ) -> None:
        """
        Constructor of the class
        l_shape : Denses layers shape of the network
        dropout: Dropout applied to dense layers
        activation_functions: Activation functions applied to dense layers
        k_embedding: Dictionnary with output shape of embeddings
        dict_shape: Dictionnary with size of vocabulary for each feature
        model_name: Name of the model
        numeric_features: List of numeric features of the model
        textual_features: List of textual features of the model to embed
        ohe_features: List of textual features of the model to one hot encode
        image: If images need to be integrated into models,
        image_shape: shape of images,
        image_layers_shape: shapes of dense layers applied to images,
        returns -> None
        """
        super().__init__()

        self.l_shape = l_shape
        self.dropout = dropout
        self.k_embedding = k_embedding
        self.embeddings = {}
        self.dict_shape = dict_shape
        self.outputs_shape = outputs_shape
        self.model_name = model_name
        self.activation_functions = activation_functions
        self.numeric_features = numeric_features
        self.textual_features = textual_features
        self.ohe_features = ohe_features
        self.image = image
        self.image_shape = image_shape
        self.image_layers_shape = image_layers_shape

        assert len(self.l_shape) - 1 == len(self.dropout)
        assert len(activation_functions) == len(l_shape)

    def define_inputs(self) -> None:
        """
        Define shape of inputs needed for each kind of category for the model
        """
        input_textual_dict = {}
        input_numeric_dict = {}
        input_ohe_dict = {}
        input_total = []
        if self.textual_features:
            for feature in self.textual_features:
                input_textual_dict[feature] = Input(shape=(1,), name=f"input_{feature}")
            input_total.append(input_textual_dict)
        if self.numeric_features:
            for feature in self.numeric_features:
                input_numeric_dict[feature] = Input(shape=(1,), name=f"input_{feature}")
            input_total.append(input_numeric_dict)
        if self.ohe_features:
            for feature in self.ohe_features:
                input_ohe_dict[feature] = Input(shape=(1,), name=f"input_{feature}")
            input_total.append(input_ohe_dict)
        if self.image:
            in_img = {}
            in_img["img"] = Input(
                shape=(self.image_shape),
                name=f"input_image",
            )
            input_total.append(in_img)
        self.input = input_total

    def set_embedding(self) -> None:
        """
        Prepare embedding layers for model
        """
        if not self.k_embedding:
            self.k_embedding = {feature: 20 for feature in self.textual_features}
        else:
            self.k_embedding = self.k_embedding
        assert len(self.k_embedding.keys()) == len(self.textual_features)
        for feature in self.textual_features:
            self.embeddings[feature] = Flatten(name=f"Flatten_{feature}")(
                Embedding(
                    input_dim=self.dict_shape[feature],
                    output_dim=self.k_embedding[feature],
                    input_length=(1,),
                    name=f"Embedding_{feature}",
                    embeddings_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                )(self.input[0][feature])
            )

    def set_core(self) -> None:
        """
        Set Dense layers for the model
        """
        if self.image:
            img = self.input[-1]
            # print(img)
            img = img["img"]
            # print(img)
            ximg = Flatten(
                # batch_input_shape=(
                #     None,
                #     self.image_shape[0],
                #     self.image_shape[1],
                #     self.image_shape[2],
                # ),
                name="Flatten_image",
            )(img)
            # print(img)
            # img = BatchNormalization()(img)
            # img = img
            for i, shape in enumerate(self.image_layers_shape):
                ximg = Dense(shape, name=f"Dense_layer_img_{i}", activation="relu")(
                    ximg
                )
            # self.input.append({"img": img})
            # self.input = self.input[:-1]
            concattedbis = Concatenate(name="concatted")(
                list(self.embeddings.values())
                + [elem for inpt in self.input[1:-1] for elem in inpt.values()],
            )
            concatted = Concatenate(name="concatted2")([concattedbis, ximg])
        else:
            concatted = Concatenate(name="concatted")(
                list(self.embeddings.values())
                + [elem for inpt in self.input[1:] for elem in inpt.values()]
            )
        dense_layers = [Dense(self.l_shape[0], name="Dense0")(concatted)]
        dropout_layers = []
        for i, layer_shape in enumerate(self.l_shape[1:]):
            dropout_layers.append(
                Dropout(self.dropout[i], name=f"Dropout{i}")(dense_layers[i])
            )
            dense_layers.append(
                Dense(
                    layer_shape,
                    name=f"Dense{i+1}",
                    activation=self.activation_functions[i],
                )(dropout_layers[i])
            )
        self.concatted = concatted
        self.dense_layers = dense_layers
        self.dropout_layers = dropout_layers

        self.outputs = Dense(self.outputs_shape, name="output")(dropout_layers[-1])

    def model(self) -> None:
        """
        Create the model from the configuration and compile it
        """
        self.define_inputs()
        self.set_embedding()
        self.set_core()
        m = Model(inputs=self.input, outputs=self.outputs, name=self.model_name)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.08)
        m.compile(
            optimizer=Adam(learning_rate=0.01),
            loss="mse",
            metrics=["mae"],
            # run_eagerly=True,
        )
        return m


######################## DEC ########################
# All the following part is very mainly taken from the repository https://github.com/zhoushengisnoob/DeepClustering, one might surely integrate directly the previous repository to use the model


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    """

    def __init__(self, n_clusters: int, weights=None, alpha=1.0, **kwargs):
        """
        Constructor of the class
        n_clusters: number of clusters.
        weights: Initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
        """
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer="glorot_uniform",
            name="clusters",
        )
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs) -> np.array:
        """
        student t-distribution, as same as used in t-SNE algorithm.
                q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.

        inputs: Input from which to compute student t distribution
        returns -> q: student's t-distribution, or soft labels for each sample
        """
        q = 1.0 / (
            1.0
            + (
                K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2)
                / self.alpha
            )
        )
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q


class DEC(object):
    """
    The aim of this class to build Deep Embedding Clustering to improve performances fo clustering
    """

    def __init__(
        self,
        dims=None,
        l_shape: List[int] = [64, 256],
        dropout=0.2,
        encode_shape: int = 5,
        activation_functions: List[str] = ["relu", "relu"],
        n_clusters=6,
    ) -> None:
        self.l_shape = l_shape
        self.activation_functions = activation_functions
        self.dims = dims
        self.n_clusters = n_clusters
        self.encode_shape = encode_shape
        self.dropout = 0.2

    def encoder_models(self):
        x = Input(shape=(self.dims[0],), name="input_encoder")
        y = x
        for i, shape in enumerate(self.l_shape):
            y = Dense(shape, name=f"encoder_layer{i}")(y)
            y = Dropout(self.dropout)(y)

        y = Dense(self.encode_shape)(y)

        output_encoder = y

        self.l_shape.reverse()
        for i, shape in enumerate(self.l_shape):
            y = Dense(shape, name=f"decoder_layer{i}")(y)
            y = Dropout(self.dropout)(y)
        self.l_shape.reverse()
        y = Dense(self.dims[0], name="output")(y)
        output_AE = y

        self.model_AE = Model(inputs=x, outputs=output_AE, name="model_AE")
        self.model_Encoder = Model(
            inputs=x, outputs=output_encoder, name="model_Encoder"
        )
        clustering_layer = ClusteringLayer(
            n_clusters=self.n_clusters, name="clustering"
        )(output_encoder)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        self.model_AE.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        self.model = Model(inputs=x, outputs=clustering_layer)

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer="sgd", loss="kld"):
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer, loss=loss)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    def fit(
        self,
        x,
        n_clusters,
        maxiter=2e4,
        batch_size=256,
        tol=1e-3,
        update_interval=1000,
    ):
        # Initialize model with Kmeans
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(self.model_Encoder(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name="clustering").set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            # print(ite)
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)
                # evaluate the clustering performance
                y_pred = q.argmax(1)

                # check stop criterion
                delta_label = (
                    np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                )
                y_pred_last = np.copy(y_pred)
                print(f"Loss value:{loss} for iteration {ite}")
                if ite > 0 and delta_label < tol:
                    print("delta_label ", delta_label, "< tol ", tol)
                    print("Reached tolerance threshold. Stopping training.")
                    break

            # train on batch
            idx = index_array[
                index * batch_size : min((index + 1) * batch_size, x.shape[0])
            ]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            ite += 1
        q = self.model.predict(x, verbose=0)
        p = self.target_distribution(q)
        return y_pred, p, q, x
