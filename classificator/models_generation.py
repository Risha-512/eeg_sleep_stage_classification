from tensorflow.keras import activations, models, optimizers, losses
from tensorflow.python.keras.layers import Input, Convolution1D, MaxPool1D, SpatialDropout1D, GlobalMaxPool1D, \
    Dropout, Dense, TimeDistributed

from common.edf_parameters import STAGES_TYPES_COUNT


class ModelCNN:
    def __init__(self):
        self.__kernel_size = 3
        self.__padding_valid = 'valid'
        self.__padding_same = 'same'
        self.__dropout_rate = 0.01
        self.__metrics = ['acc']

    def __generate_base_model(self):
        """
        Сгенерировать основную (базовую) модель

        :return: основная модель
        """
        sequence_input = Input(shape=(3000, 1))

        sequence = Convolution1D(filters=32,
                                 kernel_size=self.__kernel_size,
                                 padding=self.__padding_valid,
                                 activation=activations.relu)(sequence_input)
        sequence = Convolution1D(filters=32,
                                 kernel_size=self.__kernel_size,
                                 padding=self.__padding_valid,
                                 activation=activations.relu)(sequence)

        for filters in [32, 32, 256]:
            sequence = MaxPool1D(pool_size=2)(sequence)
            sequence = SpatialDropout1D(rate=self.__dropout_rate)(sequence)
            sequence = Convolution1D(filters=filters,
                                     kernel_size=self.__kernel_size,
                                     padding=self.__padding_valid,
                                     activation=activations.relu)(sequence)
            sequence = Convolution1D(filters=filters,
                                     kernel_size=self.__kernel_size,
                                     padding=self.__padding_valid,
                                     activation=activations.relu)(sequence)
        # финальный блок
        sequence = GlobalMaxPool1D()(sequence)
        sequence = Dropout(rate=self.__dropout_rate)(sequence)

        model = models.Model(inputs=sequence_input,
                             outputs=Dropout(rate=self.__dropout_rate)(Dense(units=64,
                                                                             activation=activations.relu)(sequence)))

        model.compile(optimizer=optimizers.Adam(),
                      loss=losses.sparse_categorical_crossentropy,
                      metrics=self.__metrics)
        return model

    def generate_cnn_model(self):
        """
        Сгенерировать модель сверточной нейронной сети

        :return: модель сверточной нейронной сети
        """
        sequence_input = Input(shape=(None, 3000, 1))

        sequence = TimeDistributed(self.__generate_base_model())(sequence_input)
        sequence = Convolution1D(filters=128,
                                 kernel_size=self.__kernel_size,
                                 padding=self.__padding_same,
                                 activation=activations.relu)(sequence)
        sequence = SpatialDropout1D(rate=self.__dropout_rate)(sequence)
        sequence = Convolution1D(filters=128,
                                 kernel_size=self.__kernel_size,
                                 padding=self.__padding_same,
                                 activation=activations.relu)(sequence)
        sequence = Dropout(rate=self.__dropout_rate)(sequence)  # 0.05

        model = models.Model(inputs=sequence_input,
                             outputs=Convolution1D(filters=STAGES_TYPES_COUNT,
                                                   kernel_size=self.__kernel_size,
                                                   padding=self.__padding_same,
                                                   activation=activations.softmax)(sequence))

        model.compile(optimizer=optimizers.Adam(),
                      loss=losses.sparse_categorical_crossentropy,
                      metrics=self.__metrics)
        model.summary()
        return model
