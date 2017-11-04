import tensorflow as tf

NUM_CLASSES = 10

def _get_tn_var(name, shape, stddev, reg=None):
    '''
    Get a variable with truncated normal initializer and optional l2 regularization. 
    '''
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

    if reg is not None:
        weight_penalty = tf.multiply(tf.nn.l2_loss(var), reg, name='weight_loss')
        tf.add_to_collection('losses', weight_penalty)
        
    return var

def _get_kernel(name, shape, stddev, reg=None):
    '''
    Alias for _get_tn_var. 
    '''
    return _get_tn_var(name, shape, stddev, reg=reg)

def capsnet(inputs):
    '''
    Construct a 3-layer capsule net with 28x28 inputs. 

    Layer 1 is a regular convulution. We blow 1 channel up into 256 channels.

    Layer 2 is the first capsule layer. It amounts to 32 parallel convolutions from 256 channels
    down to 8 channels. Each of these 32 conv layers contains (width) * (height) capsules of length 8.
    The output of the layer is a [width * height * 32] * 8 matrix. Each of the [width * height * 32] rows
    represents a capsule. 

    Layer 3 is the second capsule layer. Layer 3 capsules receive input from each capsule in layer 2. Layer 
    2 outputs are first multiplied by a weight matrix, then scaled by coupling coefficients as part of the
    routing process out of layer 2. 
    '''

    # Layer 1 - Regular Conv
    with tf.variable_scope('conv1') as scope:
        kernel = _get_kernel('weights', [9, 9, 1, 256], stddev=5e-2, reg=0.0)
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
        pre_act = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_act, name=scope.name)

    # Layer 2 - Primary Capsules
    capsules1 = []
    
    with tf.variable_scope('primary_caps') as scope:
        for i in range(0, 32):
            kernel = _get_kernel('weights' + str(i), [9, 9, 256, 8], stddev=5e-2, reg=0.0)
            biases = tf.get_variable('biases' + str(i), [8], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='VALID')
            pre_act = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_act, name=scope.name)
            shaped = tf.reshape(conv2, [36, 8])
            capsules1.append(shaped)

    capsules1 = tf.reshape(tf.stack(capsules1), (32 * 36, 8))

    # Compute routing priors for Caps1 -> Caps2 layer connections
    with tf.variable_scope('coupling') as scope:
        priors = tf.get_variable('priors', shape=[capsules1.shape[0], 10], initializer=tf.constant_initializer(0.0))
        coupling_coeffs = tf.nn.softmax(priors)


    # Now connect Caps1 -> Caps2 layers. 
    capsules2 = []
    
    with tf.variable_scope('secondary_caps'):
        for j in range(0, NUM_CLASSES):
            print(j)
            routes_into_j = []
            for i in range(0, capsules1.shape[0]):
                W_ij = _get_tn_var('weights_' + str(i) + str(j), shape=[16, 8], stddev=0.04, reg=0.004)
                b_ij = tf.get_variable('biases_' + str(i) + str(j), [16], initializer=tf.constant_initializer(0.0))
                ij_routing = tf.einsum('ij,j->i', W_ij, capsules1[i])
                uhat = tf.add(ij_routing, b_ij)
                routes_into_j.append(tf.scalar_mul(coupling_coeffs[i, j], uhat))

            sj = tf.reduce_sum(routes_into_j, 0)
            norm_sj = tf.norm(sj, ord=2)
            norm_sj_squared = tf.square(norm_sj)
            vj = tf.multiply(tf.sigmoid(norm_sj_squared), tf.divide(sj, norm_sj))
            capsules2.append(vj)

            ## TODO: Update priors
        capsules2 = tf.stack(capsules2)
    return capsules2

        