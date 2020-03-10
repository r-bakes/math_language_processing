import os
import definitions as d
import numpy as np

for subdir in os.listdir(os.path.join(d.ROOT_DIR, 'data')):
    print(subdir)
    for file in os.listdir(os.path.join(d.ROOT_DIR, 'data', subdir)):
        file_dir = os.path.join(d.ROOT_DIR, 'data', subdir, file)

        data = open(file_dir, 'r').read().splitlines()
        data = np.reshape(data, (-1, 2))
        print(data[0])



        #
        # if subdir == 'extrapolate':
        #
        # elif subdir == 'interpolate':
        #
        # elif subdir == 'train-easy':
        #
        # elif subdir == 'train-medium':
        #
        # elif subdir == 'train-hard':



# train_easy = open(os.path.join(self.dir_data, r"train-easy", self.question_type), 'r').read().splitlines()
# test = open(os.path.join(self.dir_data, r"interpolate", self.question_type), 'r').read().splitlines()
#
# train_easy = np.reshape(train_easy, (-1, 2))
# test = np.reshape(test, (-1, 2))
#
# train_x, train_y = train_easy[:, 0], train_easy[:, 1]
# train_y = np.char.add(np.full(shape=len(train_y), fill_value='\t'), train_y)
# train_y = np.char.add(train_y, np.full(shape=len(train_y), fill_value='\n'))
#
# test_x, test_y = test[:, 0], test[:, 1]
# test_y = np.char.add(np.full(shape=len(test_y), fill_value='\t'), test_y)
# test_y = np.char.add(test_y, np.full(shape=len(test_y), fill_value='\n'))
#
# data_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
# data_test = tf.data.Dataset.from_tensor_slices((test_x, test_y))