from tensorflow.keras import activations, models, optimizers, losses
from tensorflow.python.keras.layers import Input, Convolution1D, MaxPool1D, SpatialDropout1D, GlobalMaxPool1D, \
    Dropout, Dense, TimeDistributed


class ModelCNN:
    def __init__(self, classes_number, kernel_size: int = 3, pool_size: int = 2, dropout_rate: float = 0.01):
        self.__classes_number = classes_number
        self.__kernel_size = kernel_size
        self.__pool_size = pool_size
        self.__dropout_rate = dropout_rate

        self.__padding_valid = 'valid'
        self.__padding_same = 'same'
        self.__metrics = ['acc']

    def __generate_base_model(self):
        """
        Сгенерировать основную (базовую) модель

        Над исходным вектором дважды применяется сверточный слой.
        После этого происходит максимальное объединение в пул и отсеивание.
        Весь процесс повторяется 4 раза.

        :return: основная модель
        """
        sequence_input = Input(shape=(3000, 1))

        # дважды применить сверточный слой
        sequence = Convolution1D(filters=32,
                                 kernel_size=self.__kernel_size,
                                 padding=self.__padding_valid,
                                 activation=activations.relu)(sequence_input)
        sequence = Convolution1D(filters=32,
                                 kernel_size=self.__kernel_size,
                                 padding=self.__padding_valid,
                                 activation=activations.relu)(sequence)

        for filters in [32, 32, 256]:
            # применить слои максимального объедининеия в пул и отсеивания
            sequence = MaxPool1D(pool_size=self.__pool_size)(sequence)
            sequence = SpatialDropout1D(rate=self.__dropout_rate)(sequence)

            # повторно дважды применить сверточный слой
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

        # примененить полносвязный слой
        sequence = Dense(units=64, activation=activations.relu)(sequence)

        # применить последнее отбрасывание и сгенерировать модель
        model = models.Model(inputs=sequence_input,
                             outputs=Dropout(rate=self.__dropout_rate)(sequence))

        # настроить модель для обучения
        model.compile(optimizer=optimizers.Adam(),
                      loss=losses.sparse_categorical_crossentropy,
                      metrics=self.__metrics)
        return model

    def generate_cnn_model(self):
        """
        Сгенерировать модель сверточной нейронной сети

        Над вектором, полученным из базовой модели, два раза применяется сверточный слой вместе с отбрасываем.
        Затем полученный вектор подается на еще один сверточный слой.

        :return: модель сверточной нейронной сети
        """
        sequence_input = Input(shape=(None, 3000, 1))

        # применить сверточный слой и отбрасывание [1]
        sequence = TimeDistributed(self.__generate_base_model())(sequence_input)
        sequence = Convolution1D(filters=128,
                                 kernel_size=self.__kernel_size,
                                 padding=self.__padding_same,
                                 activation=activations.relu)(sequence)
        sequence = SpatialDropout1D(rate=self.__dropout_rate)(sequence)

        # применить сверточный слой и отбрасывание [2]
        sequence = Convolution1D(filters=128,
                                 kernel_size=self.__kernel_size,
                                 padding=self.__padding_same,
                                 activation=activations.relu)(sequence)
        sequence = Dropout(rate=self.__dropout_rate)(sequence)

        # примененить последний сверточный слой и сгенерировать модель
        model = models.Model(inputs=sequence_input,
                             outputs=Convolution1D(filters=self.__classes_number,
                                                   kernel_size=self.__kernel_size,
                                                   padding=self.__padding_same,
                                                   activation=activations.softmax)(sequence))

        # настроить модель для обучения
        model.compile(optimizer=optimizers.Adam(),
                      loss=losses.sparse_categorical_crossentropy,
                      metrics=self.__metrics)
        model.summary()
        return model
