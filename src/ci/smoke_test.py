import numpy as np
import tensorflow as tf

def main():
    BATCH = 2
    IMAGE_SIZE = 416
    GRID = 13
    NUM_CLASSES = 3
    OUTPUT_CH = 5 + NUM_CLASSES

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(OUTPUT_CH, 1, activation=None),
        tf.keras.layers.Reshape((GRID, GRID, OUTPUT_CH))
    ])

    model.compile(optimizer='adam', loss='mse')

    x = np.random.rand(BATCH, IMAGE_SIZE, IMAGE_SIZE, 3).astype('float32')
    y = np.random.rand(BATCH, GRID, GRID, OUTPUT_CH).astype('float32')

    loss = model.train_on_batch(x, y)
    print('smoke test training loss:', loss)


if __name__ == '__main__':
    main()
