import tensorflow as tf

def create_backtrack_array(trading_data, backtrack):
    """"
    creates a 2-D array backtracking the time specified in
    parameter backtrack.
    """
    import numpy as np
    array_length = len(trading_data)
    array_width = backtrack
    backtrack_array = np.zeros([array_length, array_width])

    for i in range(0, array_width):
        for j in range(0, array_width):
            backtrack_array[i, j] = trading_data[i]

    for i in range(array_width, array_length):
        for j in range(0, array_width):
            backtrack_array[i, j] = trading_data[i + j - array_width]

    return backtrack_array

def create_difference_array(trading_data):
    """
    creates a 0-D array of the difference in fractional 0.1 PIPs between each value

    :param trading_data:
    :return:
    """
    import numpy as np
    array_length = len(trading_data)
    diff_array = np.zeros(array_length)
    for i in range(1, array_length):
        diff_array[i] = int((trading_data[i] - trading_data[i - 1]) * 100000)

    return diff_array

def create_forward_max(trading_data, forwardtracking):
    """
    creates a labelling matrix for the maximum of the next X values specified in --forwardtracking--
    :param trading_data:
    :param forwardtracking:
    :return:
    """
    import numpy as np

    backtrack = forwardtracking
    array_length = len(trading_data)
    label_max = np.zeros([array_length, 1])

    backtrack_array = create_backtrack_array(trading_data, backtrack)
    amax = np.amax(backtrack_array, axis=1)

    ## amax is the maximum of the 1-D x array each row that represents the last --30--(maybe change to a variable amount) minutes/values
    ## so.. amax is basically the max of the LAST 30 mins
    ## label_max is the same array but shifted 30 backwards, so as to map each array of values to the amax of 30 minutes ahead

    for i in range(0, array_length - backtrack):
        label_max[i, 0] = amax[i + backtrack]

    for i in range(array_length - backtrack, array_length):
        label_max[i, 0] = amax[i]

    return label_max


def create_forward_maxdiff(trading_data, forwardtracking):

    import numpy as np

    backtrack = forwardtracking
    array_length = len(trading_data)
    label_maxdiff = np.zeros([array_length, 1])

    backtrack_array = create_backtrack_array(trading_data, backtrack)
    amax = np.amax(backtrack_array, axis=1)

    for i in range(0, array_length - backtrack - 1):
        label_maxdiff[i, 0] = int((amax[i + backtrack +1] - trading_data[i])*100000)

    for i in range(array_length - backtrack - 1, array_length):
        label_maxdiff[i, 0] = 0


    return label_maxdiff


def get_stochastic_batch(timestamp, set, labels, size = 20):
    import numpy as np
    # print(len(set))
    # print(len(np.unique(set)))
    # print(len(np.unique(labels)))
    indices = np.random.choice(len(set), size, replace= False)
    batch_stamps = np.take(timestamp,indices, axis=0)
    batch_set = np.take(set,indices, axis=0)
    batch_labels = np.take(labels, indices, axis=0)
    # print(batch_labels)

    return batch_stamps, batch_set, batch_labels


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)
    return


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, 'activation')
        tf.histogram_summary(layer_name + '/activations', activations)
    return activations

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



def plot_clusters(all_samples, centroids, n_samples_per_cluster):
    import matplotlib.pyplot as plt
    # Plot out the different clusters
    # Choose a different colour for each cluster
    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i, centroid in enumerate(centroids):
        # Grab just the samples fpr the given cluster and plot them out with a new colour
        samples = all_samples[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
        plt.scatter(samples[:,0], samples[:,1], c=colour[i])
        # plt.pause(0.000001)
        # Also plot centroid
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
        # plt.pause(0.000001)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
        # plt.pause(0.000001)
    plt.pause(0.0000001)

    return