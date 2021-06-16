import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # использование CPU (без использования GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # отключение лога


from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
import keras
from collections import Counter
import numpy as np
import time
from tkinter import *
from functools import partial


class Vectorizer:
    """
    класс для преобразования текста в векторы целых чисел и наоборот
    """

    def __init__(self, text):
        tokens = text.lower()  # т.к. работаем с символами, достаточно привести текст к одному регистру
        tokens_counts = Counter(tokens)  # подкласс словаря (Counter) с парами 'символ':счетчик
        # создаем отсортированный по убыванию счетчика список символов (токенов)
        tokens = [x[0] for x in tokens_counts.most_common()]
        self._token_to_ind = {x: i for i, x in enumerate(tokens)}  # создаем словарь токен:индекс
        self._ind_to_token = {i: x for i, x in enumerate(tokens)}  # создаем словарь индекс:токен
        self.vocab_size = len(tokens)  # размер словаря == количество токенов

    def vectorize(self, text):
        """
        векторизация текста
        """
        tokens = text.lower()
        indices = []
        for token in tokens:
            if token in self._token_to_ind:
                indices.append(self._token_to_ind[token])
        return np.array(indices, dtype=np.int32)

    def unvectorize(self, vector):
        """
        де-векторизация текста
        """
        # преобразуем вектор индексов в список символов
        tokens = [self._ind_to_token[index] for index in vector]
        return ''.join(tokens)  # т.к. токенами выступают символы - собираем строку без доп. разделителей


def _create_sequences(vector, seq_length, seq_step):
    """
    делит вектор длиной seq_length на массив векторов длиной seq_step
    """
    passes = []
    for offset in range(0, seq_length, seq_step):
        pass_samples = vector[offset:]
        num_pass_samples = pass_samples.size // seq_length
        pass_samples = np.resize(pass_samples, (num_pass_samples, seq_length))
        passes.append(pass_samples)
    return np.concatenate(passes)


def shape_for_stateful_rnn(data, batch_size, seq_length, seq_step):
    """
    разделение вектора на входные и целевые данные
    data:   abcd
    input:  abc
    target: bcd
    """
    inputs = data[:-1]
    targets = data[1:]
    # деление по размеру seq_step
    inputs = _create_sequences(inputs, seq_length, seq_step)
    targets = _create_sequences(targets, seq_length, seq_step)
    # коррекция по батчу
    inputs = _batch_sort_for_stateful_rnn(inputs, batch_size)
    targets = _batch_sort_for_stateful_rnn(targets, batch_size)
    # приведение target к трехмерному массиву
    targets = targets[:, :, np.newaxis]
    return inputs, targets


def _batch_sort_for_stateful_rnn(sequences, batch_size):
    """
    обеспечивает стабильную работу при любом батче
    """
    num_batches = sequences.shape[0] // batch_size
    num_samples = num_batches * batch_size
    reshuffled = np.zeros((num_samples, sequences.shape[1]), dtype=np.int32)
    for batch_index in range(batch_size):
        slice_start = batch_index * num_batches
        slice_end = slice_start + num_batches
        index_slice = sequences[slice_start:slice_end, :]
        reshuffled[batch_index::batch_size, :] = index_slice
    return reshuffled


def load_data(data_file, batch_size, seq_length, seq_step):
    global vectorizer

    with open(data_file, encoding='utf-8') as input_file:
        text = input_file.read()

    vectorizer = Vectorizer(text)
    data = vectorizer.vectorize(text)  # получение векторов
    x, y = shape_for_stateful_rnn(data, batch_size, seq_length, seq_step)

    return x, y, vectorizer


def make_model(batch_size, vocab_size, embedding_size, rnn_size, num_layers):
    """
    создание слоев Sequenrial модели
    """
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, batch_input_shape=(batch_size, None)))
    for layer in range(num_layers):
        model.add(LSTM(rnn_size, stateful=True, return_sequences=True))
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def train(model, x, y, batch_size, num_epochs):
    """
    обучение модели
    """
    print("Обучение...")
    train_start = time.time()
    callbacks = None
    model.fit(x, y, batch_size=batch_size,
              shuffle=False,
              epochs=num_epochs,
              verbose=1,
              callbacks=callbacks)
    train_end = time.time()
    print(f"Время обучения: {(train_end - train_start) // 60} минут")


def sample_preds(preds, temperature=0.5):
    """
    выбор вероятности, параметр temperature влияет на разброс
    """
    preds = np.asarray(preds).astype(np.float64)  # получаем массив вероятностей в float64
    preds += np.finfo(np.float64).tiny  # прибавляем float64.tiny к каждому члену - недопускаем log(0)
    exp_preds = np.exp(np.log(preds) / temperature)
    probas = np.random.multinomial(1, exp_preds / np.sum(exp_preds), 1)  # выбор значений исходя из вероятностей
    return np.argmax(probas)


def generate(model, vectorizer, seed, length, diversity=0.5):
    """
    Генерация строки. temperature=0.5
    """
    seed_vector = vectorizer.vectorize(seed)  # семя в вектор
    model.reset_states()  # обнуление состояний (метрик)
    # предсказание
    preds = None
    for char_index in np.nditer(seed_vector):
        preds = model.predict(np.array([[char_index]]), verbose=0)

    sampled_indices = []
    # генерация по одному символу (токену)
    for i in range(length):
        char_index = 0
        if preds is not None:  # если есть предсказание
            char_index = sample_preds(preds[0][0], diversity)  # выбрать символ (индекс вектора)
        sampled_indices.append(char_index)  # добавить символ (вектор) к выводу
        preds = model.predict(np.array([[char_index]]), verbose=0)  # новое предсказание
    sample = seed + vectorizer.unvectorize(sampled_indices)  # девекторизация списка символов (векторов)
    return sample


def button_click(vectorizer, embedding_size, rnn_size, num_layers, text):
    predict_model = make_model(1, vectorizer.vocab_size, embedding_size, rnn_size, num_layers)
    predict_model.set_weights(model.get_weights())
    res = generate(predict_model, vectorizer, seed=seed, length=out_len).split("\n\n")

    text.insert(1.0, "\n\n".join(res[1:-1]) + "\n\n")


if __name__ == "__main__":
    # настройки ###
    # batch size - количество образцов на одно обновление градиента
    # больше батч => меньше итераций на эпоху (быстрее обучение), но потребляет больше оперативной памяти
    # при текущих параметрах рост негативно влияет на качество обучения (выбран наибольший размер с уд. скоростью)
    batch_size = 4
    num_epochs = 25  # количество эпох обучения
    out_len = 400  # длина генерируемой последовательности
    seq_length = 128  # длина одной последовательности, которая будет использована для обучения
    # шаг начала последовательностей длины seq_length (меньше значение => точность выше, скорость меньше)
    seq_step = 4
    rnn_size = 128  # количество узлов (нейронов) одного слоя LSTM
    embedding_size = 4  # кол-во выходов слоя Embedding
    num_layers = 2  # количество слоев LSTM + Dropout
    data_file = ".\\anecs3.txt"  # файл датасета
    # файл обученной модели
    model_file = ".\\educated_model.h5"
    seed = "а"  # начальный символ
    #

    # получение объектов
    x, y, vectorizer = load_data(data_file, batch_size, seq_length, seq_step)

    try:
        print(f"Попытка загрузки модели из файла {model_file}...")
        model = keras.models.load_model(model_file)
    except:
        print(f"Файл {model_file} не найден, модель будет обучена.")
        # получение объекта модели
        model = make_model(batch_size, vectorizer.vocab_size, embedding_size, rnn_size, num_layers)
        # обучение модели
        train(model, x, y, batch_size, num_epochs)
        # сохранение модели
        model.save(filepath=model_file)

    # генерация текста
    print(f"Модель получена. Запуск интерфейса. ")

    clicked = False
    root = Tk()
    root.title("Генератор анекдотов")
    root.geometry("400x400")
    frame = Frame()
    frame.pack()
    text = Text(frame, width=400, height=300)
    button = Button(frame, text="Сгенерировать", command=partial(button_click, vectorizer, embedding_size, rnn_size, num_layers, text))
    # button.place(relx=.5, rely=.0, width=400, anchor="c")
    button.pack()
    text.pack()
    root.mainloop()
