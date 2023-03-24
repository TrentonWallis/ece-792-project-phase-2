# -*- coding: utf-8 -*-
if __name__ == '__main__':

    import pickle
    def save_pickle(thing, filename):
        with open(filename, "wb") as f: # "wb" because we want to write in binary mode
            pickle.dump(thing, f)

    import tensorflow as tf
    from tqdm import tqdm
    import compression as compress
    from multiprocessing import Pool, cpu_count
    from timeit import default_timer as timer
    import numpy as np
    import matplotlib.pyplot as plt

    cifar100 = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    (x_train, y_train), (x_test, y_test) = cifar100

    #need to change x's into y's and compressed vedions into x's
    #powers of two
    compression_steps = 16
    down_sample_ratio = 1
    source_size = 8 #Source block size
    compressed_size = 4 #destination block size

    # shape after compression (3, 16, 16, 6) => RGB, 16x16, params
    type_str = "_"+str(source_size)+"_"+str(compressed_size)+"_"+str(down_sample_ratio)+"_"+str(compression_steps)+'.dat'
    xt = x_train
    pool = Pool(cpu_count())
    print('Compressing Training Images')
    y_train = xt[5000:]
    x_train = []
    for img in y_train:
        x_train.append(pool.apply_async(compress.fractal_compress_rgb,(img,down_sample_ratio,source_size,compressed_size,compression_steps)))
    x_train = [r.get(timeout=100) for r in x_train]
    print('Saving Training Images')
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.array(y_train[2][1]).astype(np.uint8), interpolation='none')
    plt.subplot(122)
    plt.imshow(np.array(x_train[2][1]).astype(np.uint8), interpolation='none')
    plt.show()
    save_pickle([x_train,y_train], 'train'+ type_str)

    print('Compressing Validation Images')
    y_valid = xt[:5000]
    x_valid = []
    for img in y_valid:
        x_valid.append(pool.apply_async(compress.fractal_compress_rgb,(img,down_sample_ratio,source_size,compressed_size,compression_steps)))
    x_valid = [r.get(timeout=100) for r in x_valid]
    print('Saving Validation Images')
    save_pickle([x_valid,y_valid], 'valid'+ type_str)

    print('Compressing Testing Images')
    y_test = x_test
    x_test_comp = []
    for img in x_test:
        x_test_comp.append(pool.apply_async(compress.fractal_compress_rgb,(img,down_sample_ratio,source_size,compressed_size,compression_steps)))
    x_test = [r.get(timeout=100) for r in x_test_comp]
    print('Saving Test Images')
    save_pickle([x_test,y_test], 'test'+ type_str)
    pool.close()
