import numpy as np
np.set_printoptions(precision = 10, linewidth = 150, suppress = True)

from trading_functions import get_stochastic_batch, create_backtrack_array, create_difference_array, create_forward_maxdiff
from classification_training import train_classes
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, label_binarize

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


## ORIGINAL DATA
filepath = '/home/patrick/dev/hannah/data'

filename = 'EURUSD_Candlestick_1_m_ASK_01.01.2013-31.12.2013_2.txt'
full_filepath = filepath + '/' + filename

##
filename_mode = 'EURUSD_Candlestick_2013_mode'
full_filename_mode = filepath + '/' + filename_mode

##
filename_labels = 'EURUSD_Candlestick_2013_labels'
full_filename_labels = filepath + '/' + filename_labels

##
filename_binary_labels = 'EURUSD_Candlestick_2013_binary_labels'
full_filename_binary_labels = filepath + '/' + filename_binary_labels


## PARAMETERS
readout_mode = "Other"             ## possible readout modes are --Original-- and --Other--
save_mode = "Old"                     ## possible save modes are --New-- and --Old--
save_type = "npy"                     ## possible save type are --npy-- and --txt--
backtrack = 30
forwardtracking = 30
rng = 47
rng_str = str(rng)


flags.DEFINE_string('summaries_dir', filepath + '/logs', 'Directory for storing logs')
flags.DEFINE_string('variables_dir', filepath + '/variables', 'Directory for storing variable data')
# classification
flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_integer('features',8, 'Amount of features per convolutional step')

flags.DEFINE_string('train', 'train', '-train the set, -evaluate a specific entry or -both')
flags.DEFINE_boolean('anew', False, 'Continue training variables or create a new one. Will create new, if no previous data is availabe')




if readout_mode == "Original" or save_mode == "New":
    print('Reading out data from', full_filepath, '..')
    trading_data = np.genfromtxt(full_filepath, dtype = float, skip_header = 1, delimiter='\t', autostrip= True, usecols = (2))


# array_width = backtrack

## MODE
if save_mode == 'New':
    trading_mode = create_difference_array(trading_data)
    bt_mode_array = create_backtrack_array(trading_mode, backtrack).astype(np.int16)
    if save_type == 'txt':
        print('Saving txt mode data in', full_filename_mode, '..')
        np.savetxt(full_filename_mode, bt_mode_array, delimiter = '/t')
    elif save_type == 'npy':
        print('Saving npy mode data in', full_filename_mode, '..')
        np.save(full_filename_mode, bt_mode_array)
    else:
        raise ValueError('Save_type must be either -txt- or -npy-')
elif save_mode == 'Old':
    if save_type == 'txt':
        print('Reading out txt mode data from', full_filename_mode, '..')
        bt_mode_array = np.genfromtxt(full_filename_mode)
    elif save_type == 'npy':
        print('Reading npy out mode data from', full_filename_mode, '..')
        bt_mode_array = np.load(full_filename_mode + '.npy')
    else:
        raise ValueError('Save_type must be either -txt- or -npy-')
else:
    raise ValueError


## LABELS

## REGULAR
## SAVING
if save_mode == 'New':
    labels = create_forward_maxdiff(trading_data, forwardtracking).astype(np.int16)
    if save_type == 'txt':
        print('Saving txt labels data in', full_filename_labels, '..')
        np.savetxt(full_filename_labels, labels, delimiter = '/t')
    elif save_type =='npy':
        print('Saving npy labels data in', full_filename_labels, '..')
        np.save(full_filename_labels, labels)
    else:
        raise ValueError('Your save_type must be either -txt- or -npy-')
## LOADING
elif save_mode == 'Old':
    if save_type == 'txt':
        print('Reading out txt label data from', full_filename_labels, '..')
        labels = np.genfromtxt(full_filename_labels)
    elif save_type == 'npy':
        print('Reading out npy label data from', full_filename_labels, '..')
        labels = np.load(full_filename_labels + '.npy')
    else:
        raise ValueError('Save_type must be either -txt- or -npy-')
else:
    raise ValueError()

## ONEHOT
## SAVING
if save_mode == 'New':
    lb = LabelBinarizer()
    lb.fit(labels)
    label_classes = lb.classes_
    print(label_classes)

    binary_labels = label_binarize(labels, classes=label_classes).astype(np.int8)
    if save_type == 'txt':
        print('Saving txt labels data in', full_filename_binary_labels, '..')
        np.savetxt(full_filename_binary_labels, binary_labels, delimiter = '/t')
    elif save_type =='npy':
        print('Saving npy labels data in', full_filename_binary_labels, '..')
        np.save(full_filename_binary_labels, binary_labels)
    else:
        raise ValueError('Your save_type must be either -txt- or -npy-')
## LOADING
elif save_mode == 'Old':
    if save_type == 'txt':
        print('Reading out txt label data from', full_filename_binary_labels, '..')
        binary_labels = np.genfromtxt(full_filename_binary_labels)
    elif save_type == 'npy':
        print('Reading out npy label data from', full_filename_binary_labels, '..')
        binary_labels = np.load(full_filename_binary_labels + '.npy')
    else:
        raise ValueError('Save_type must be either -txt- or -npy-')
else:
    raise ValueError()



#
# print(bt_mode_array)
# # print(labels)
unique_labels = len(np.unique(labels))
# print(len(np.unique(labels)))
# print(len(np.unique(bt_mode_array)))
#
array_length = len(labels)
indices = np.arange(0,array_length)

# scaled_mode = MinMaxScaler().fit_transform(bt_mode_array)
# scaled_labels = MinMaxScaler().fit_transform(labels)

# print(scaled_mode)
# print(scaled_labels)
#
# lb = LabelBinarizer()
# lb.fit(labels)
# label_classes =lb.classes_
# # print(label_classes)
# binary_labels = label_binarize(labels, classes = label_classes)
# scaled_binary_labels = lb.fit_transform(scaled_labels)
# print(len(binary_labels))
# print(np.shape(binary_labels))
print(np.nonzero(binary_labels[1350:1370,:])[1])
# print(binary_labels[1350:1370,:])
# print(np.nonzero(binary_labels))
# print(len(np.unique(np.nonzero(binary_labels))))
# print(len(np.unique(np.sum(binary_labels))))
# print(scaled_binary_labels)
# print(len(np.unique(scaled_binary_labels)))

train_timestamps, test_timestamps, train_set, test_set, train_labels, test_labels = train_test_split(indices, bt_mode_array,
                                                                                         binary_labels, train_size=0.5,
                                                                                         random_state=rng)


print('Train set:', len(train_set[:,0]), 'entries')
print('Train_set:', len(np.unique(np.nonzero(train_labels)[1])), 'different labels')
print('Test set:', len(test_set[:,0]), 'entries')
print('Test_set:', len(np.unique(np.nonzero(test_labels)[1])), 'different labels')
print(train_timestamps)
print(train_set)
print(train_set[0,:])






train_classes(train_timestamps, train_set, train_labels, test_timestamps, test_set, test_labels, train_labels, test_labels, unique_labels,
              features = FLAGS.features, learning_rate = FLAGS.learning_rate, max_steps = FLAGS.max_steps, rng_str = rng_str, dropout = FLAGS.dropout,
              summaries_dir = FLAGS.summaries_dir, variables_dir = FLAGS.variables_dir, mode = FLAGS.train)
