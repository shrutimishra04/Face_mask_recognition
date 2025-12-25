import numpy as np


def main():
    # lightweight computation-only smoke test (no TF required)
    a = np.random.rand(200, 200).astype('float32')
    b = np.random.rand(200, 200).astype('float32')
    c = a.dot(b)
    s = float(np.mean(c))
    print('smoke test result:', s)


if __name__ == '__main__':
    main()
