from typing import List

from tensorflow.keras import activations, models, optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.python.keras.layers import Input, Convolution1D, MaxPool1D, SpatialDropout1D, GlobalMaxPool1D, \
    Dropout, Dense, TimeDistributed
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import Callback


class ModelCNN:
    """
    Класс для генерации модели

    Аргументы:
        - classes_number: количество классов
        - kernel_size: размер ядра свертки
        - pool_size: размер максимального пула
        - dropout_rate: уровень отсеивания
    """
    def __init__(self, classes_number, kernel_size: int = 3, pool_size: int = 2, dropout_rate: float = 0.01):
        self.__classes_number = classes_number
        self.__kernel_size = kernel_size
        self.__pool_size = pool_size
        self.__dropout_rate = dropout_rate

        self.__padding_valid = 'valid'
        self.__padding_same = 'same'
        self.__metrics = ['acc']

    def __generate_base_model(self) -> Model:
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
            sequence = MaxPool1D(pool_size=self.__pool_size, padding=self.__padding_valid)(sequence)
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

        # применить последнее отсеивание и сгенерировать модель
        model = models.Model(inputs=sequence_input,
                             outputs=Dropout(rate=self.__dropout_rate)(sequence))

        # настроить модель для обучения
        model.compile(optimizer=optimizers.Adam(),
                      loss=losses.sparse_categorical_crossentropy,
                      metrics=self.__metrics)
        return model

    def generate_cnn_model(self) -> Model:
        """
        Сгенерировать модель сверточной нейронной сети

        Над вектором, полученным из базовой модели, два раза применяется сверточный слой вместе с отбрасыванием.
        Затем полученный вектор подается на еще один сверточный слой.

        :return: модель сверточной нейронной сети
        """
        sequence_input = Input(shape=(None, 3000, 1))

        # применить сверточный слой и отсеивание [1]
        sequence = TimeDistributed(self.__generate_base_model())(sequence_input)
        sequence = Convolution1D(filters=128,
                                 kernel_size=self.__kernel_size,
                                 padding=self.__padding_same,
                                 activation=activations.relu)(sequence)
        sequence = SpatialDropout1D(rate=self.__dropout_rate)(sequence)

        # применить сверточный слой и отсеивание [2]
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
        return model


class ModelCallbacks:
    """
    Класс для генерации обратных вызовов

    Аргументы:
        - model_file_path: путь к файлу модели
        - monitor: отслеживаемый параметр (монитор); возможные значения: 'val_acc', 'val_loss'
        - mode: максимизация или минимизация монитора; возможные значения: 'max', 'min', 'auto'
        - es_patience: значение ожидания для EarlyStopping
        - rlr_patience: значение ожидания для ReduceLROnPlateau
    """
    def __init__(self,
                 model_file_path: str,
                 monitor: str = 'val_acc',
                 mode: str = 'max',
                 es_patience: int = 10,
                 rlr_patience: int = 5):
        self.__model_file_path = model_file_path
        self.__monitor = monitor
        self.__mode = mode
        self.__es_patience = es_patience
        self.__rlr_patience = rlr_patience

    def generate_model_callbacks(self) -> List[Callback]:
        """
        Сгенерировать список экземпляров обратных вызовов, применяемых во время обучения
            - ModelCheckpoint - сохраняет модель (или веса) в файл контрольной точки
            - EarlyStopping - прекращает обучение, когда отслеживаемый показатель перестал улучшаться
            - ReduceLROnPlateau - уменьшить скорость обучения, когда отслеживаемый показатель перестал улучшаться

        :return: список экземпляров обратных вызовов
        """
        checkpoint = ModelCheckpoint(filepath=self.__model_file_path,
                                     monitor=self.__monitor,
                                     mode=self.__mode,
                                     verbose=1,
                                     save_best_only=True)
        early_stopping = EarlyStopping(monitor=self.__monitor,
                                       mode=self.__mode,
                                       patience=self.__es_patience,
                                       verbose=1)
        reduce_learning_rate = ReduceLROnPlateau(monitor=self.__monitor,
                                                 mode=self.__mode,
                                                 patience=self.__rlr_patience,
                                                 verbose=1)

        return [checkpoint, early_stopping, reduce_learning_rate]
