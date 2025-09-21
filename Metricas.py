from tensorflow.keras import datasets, layers, models # type: ignore
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np

import seaborn as sns

import pandas as pd

from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    tf.__version__

    # ================= Tensorboard =================

    # %load_ext tensorboard

    logdir='log'


    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    train_images, test_images = train_images / 255.0, test_images / 255.0
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=train_images, 
              y=train_labels,
              epochs=5,
              validation_data=(test_images, test_labels))

    # ================= Métricas de avaliação =================

    y_pred = np.argmax(model.predict(test_images), axis=1)
    cm = confusion_matrix(test_labels, y_pred)

    print("Matriz de Confusão:")
    print(cm)

    TN, FP, FN, TP, *rest = cm.ravel()

    # TN = True Negative
    # FP = False Positive
    # FN = False Negative
    # TP = True Positive

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Acurácia: {accuracy:.2f}")
    print(f"Precisão: {precision:.2f}")
    print(f"Recall (Sensibilidade): {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")

    # ================= Matriz de confusão =================

    y_true = test_labels
    y_pred = np.argmax(model.predict(test_images), axis=1)

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index = classes,
                              columns = classes)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


