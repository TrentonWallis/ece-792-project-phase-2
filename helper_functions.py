import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

#for timing wrapper
from functools import wraps
import time

file_name = "train_32_16_4_8.dat"

'''
Pickled Image Structure

unpickled object = [x_images, y_iamges]

x_images = [
    [empty_list, compressed_image_0],
    [empty_list, compressed_image_1]
    [empty_list, compressed_image_2]
    .
    .
    .

]

y_images = [
    original_image_0,
    original_image_1,
    original_image_2,
    .
    .
    .
    .
]


'''

def see_images(left_image, right_image):
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.array(left_image).astype(np.uint8), interpolation='none')
    plt.subplot(122)
    plt.imshow(np.array(right_image).astype(np.uint8), interpolation='none')
    plt.show()

def unpack_pickle(path_with_file_name):
    x = y = None
    with open(path_with_file_name, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        x = data[0]
        y = data[1]
        # Need to take out empty array in x set, since there is a left over array on the left side of the list for each image (see comment at top of file)
        x = [temp[1] for temp in x]
    return x, y

### Functions for saving and loading models

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path_to_saved_model, model_class):
    model = model_class()
    model.load_state_dict(torch.load(path_to_saved_model))
    model.eval()
    return model


def timer(func):
    '''Wrap the decoding function with this to measure decoding time.'''
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


#### Testing Module


def main():
    path = "./data/"
    file_path = path + "train_32_16_4_8.dat"
    x_train, y_train = unpack_pickle(file_path)
    see_images(x_train[0], y_train[0])
    x_image_shape = x_train[0].shape
    y_image_shape = y_train[0].shape
    print("X Image Shape:", x_image_shape)
    print("Y Image Shape:", y_image_shape)

if __name__ == "__main__":
    main()






