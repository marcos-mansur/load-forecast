from typing import Dict

import tensorflow as tf
import yaml

params = yaml.safe_load(open("params.yaml"))


def create_single_shot_model(params: Dict) -> tf.keras.models.Sequential:

    neurons_for_each_layer = params["train"]["NEURONS"]
    target_period = params["featurize"]["TARGET_PERIOD"]
    layer_list = params["train"]["LAYERS"]

    assert (
        type(neurons_for_each_layer) == list
    ), 'Input "neurons" to the function create_model() must be list'

    # LSTM
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None])
    )
    model.add(tf.keras.layers.BatchNormalization())

    # add hidden layers
    for enumerator, neurons, layer in zip(
        range(len(neurons_for_each_layer)), neurons_for_each_layer, layer_list
    ):

        # return_sequences must be True to hidden layers except for the last one
        if enumerator + 1 == len(neurons_for_each_layer):
            return_sequences = False
        else:
            return_sequences = True

        if layer == "LSTM":
            model.add(
                tf.keras.layers.LSTM(
                    neurons, return_sequences=return_sequences, activation="tanh"
                )
            )
        elif layer == ["RNN"]:
            model.add(
                tf.keras.layers.SimpleRNN(
                    neurons, return_sequences=return_sequences, activation="tanh"
                )
            )

    model.add(tf.keras.layers.Dense(target_period))
    model.add(tf.keras.layers.Lambda(lambda x: x * 10000.0))

    return model


#  builds an autoregressive RNN model that outputs a single time step
class AutoregressiveModel(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.reshape_layer = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]
        )
        self.norm_layer = tf.keras.layers.BatchNormalization()
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(1)  # number of features
        self.scale_layer = tf.keras.layers.Lambda(lambda x: x * 10000.0)

    def warmup(self, inputs):
        """
        This model needs is a warmup method to initialize its internal state
        based on the inputs. Once trained, this state will capture the
        relevant parts of the input history. This is equivalent to the
        single-step LSTM model.

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)

        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        """
        With the RNN's state, and an initial prediction you can now continue
        iterating the model feeding the predictions at each step back as the input.

        Args:
            inputs (_type_): _description_
            training (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        inputs = self.reshape_layer(inputs)
        inputs = self.norm_layer(inputs)

        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        predictions = self.scale_layer(predictions)

        return predictions


def create_model(params):

    if params["featurize"]["MODEL_TYPE"] == "AUTOREGRESSIVE":
        return AutoregressiveModel(
            units=params["train"]["NEURONS"][0],
            out_steps=params["featurize"]["TARGET_PERIOD"],
        )

    if params["featurize"]["MODEL_TYPE"] == "SINGLE-SHOT":
        return create_single_shot_model(params=params)
