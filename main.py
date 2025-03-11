import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Параметры
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
num_epochs = 100

# Путь к директориям
positive_dir = 'data/positive'
negative_dir = 'data/negative'
neutral_dir = 'data/neutral'
mixed_dir = 'data/mixed'  # Директория для смешанных текстов
pos_mixed_dir = 'data/pos_mixed'  # Директория для текстов с положительным и нейтральным аспектом

# Функция загрузки данных
def load_data(directory):
    texts = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

# Загрузка данных
positive_texts = load_data(positive_dir)
negative_texts = load_data(negative_dir)
neutral_texts = load_data(neutral_dir)
mixed_texts = load_data(mixed_dir)
pos_mixed_texts = load_data(pos_mixed_dir)

# Объединение текстов и создание меток
texts = positive_texts + negative_texts + neutral_texts + mixed_texts + pos_mixed_texts
labels = ([1] * len(positive_texts) +    # 1: положительный
          [0] * len(negative_texts) +    # 0: отрицательный
          [2] * len(neutral_texts) +     # 2: нейтральный
          [3] * len(mixed_texts) +       # 3: положительный с негативным аспектом
          [4] * len(pos_mixed_texts))    # 4: положительный с нейтральным аспектом

# Токенизация текста
tokenizer = Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# Преобразование текстов в последовательности
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Преобразование меток в numpy массив
labels = np.array(labels)

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # Изменено на 5 классов
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(padded, labels, epochs=num_epochs, verbose=2)

# Создание обратного словаря для преобразования индексов в слова
reverse_word_index = {index: word for word, index in word_index.items()}

# Функция для классификации текста
def classify_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    prediction = model.predict(padded_sequence)[0]
    
    class_names = ["Отрицательный", "Положительный", "Нейтральный", 
                   "Положительный с негативным аспектом", "Положительный с нейтральным аспектом"]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    
    return f"Классификация текста: {class_names[class_index]} (уверенность: {confidence:.2f})"

# Функция для генерации текста по метке
def generate_text(label, start_string, num_words=50):
    input_text = start_string
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([input_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        
        predictions = model.predict(padded_sequence, verbose=0)
        label_probabilities = predictions[0]
        
        if np.argmax(label_probabilities) == label:
            predicted_index = np.argmax(label_probabilities)
        else:
            predicted_index = np.argmax(predictions)
        
        word = reverse_word_index.get(predicted_index, "")
        if word == "":
            break
        
        input_text += " " + word

    return input_text

# Пример использования
label_dict = {
    "Отрицательный": 0,
    "Положительный": 1,
    "Нейтральный": 2,
    "Положительный с негативным аспектом": 3,
    "Положительный с нейтральным аспектом": 4
}

# Классификация текста
user_input = input("Введите текст для классификации: ")
classification_result = classify_text(user_input)
print(classification_result)

# Генерация текста
selected_label = input('Настроение гененрируемого текста:')
start_string = " "  # Начальная строка
generated_text = generate_text(label_dict[selected_label], start_string, num_words=50)
print(f"Сгенерированный текст для метки '{selected_label}': {generated_text}")
