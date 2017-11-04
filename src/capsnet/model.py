import tensorflow as tf

NUM_CLASSES = 10
M_PLUS = 0.9
M_MINUS = 0.1
LAMBDA = 0.5

INITIAL_LEARNING_RATE = 0.1
DECAY_STEPS = 100000
LEARNING_RATE_DECAY_FACTOR = 0.96


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
            inputs_to_j = []

            for i in range(0, capsules1.shape[0]):
                W_ij = _get_tn_var('weights_' + str(i) + str(j), shape=[16, 8], stddev=0.04, reg=0.004)
                b_ij = tf.get_variable('biases_' + str(i) + str(j), [16], initializer=tf.constant_initializer(0.0))
                uhat_ji = tf.add(tf.einsum('ij,j->i', W_ij, capsules1[i]), b_ij)
                inputs_to_j.append(uhat_ji)

            inputs_to_j = tf.stack(inputs_to_j)
            routed_inputs = tf.transpose(tf.multiply(tf.transpose(inputs_to_j), coupling_coeffs[:, j]))
            s_j = tf.reduce_sum(routed_inputs, 0)

            norm_s_j = tf.norm(s_j, ord=2)
            norm_s_j_sq = tf.square(norm_s_j)
            v_j = tf.multiply(tf.sigmoid(norm_s_j_sq), tf.divide(s_j, norm_s_j))

            capsules2.append(tf.norm(v_j, ord=2))

            for i in range(0, capsules1.shape[0]):
                tf.add_to_collection('prior_updates_ij', tf.einsum('k,k->', v_j, inputs_to_j[i]))
                
    return tf.stack(capsules2)

def margin_loss(targets, capsules):
    n = targets.shape[0]
    zeros, ones = tf.zeros([n]), tf.ones([n])
    m_plus, m_minus = tf.fill([n], M_PLUS), tf.fill([n], M_MINUS)

    max_0 = tf.square(tf.maximum(zeros, m_plus - capsules))
    max_1 = tf.square(tf.maximum(zeros, capsules - m_minus))

    summand0 = tf.einsum('i,i->', targets, max_0)
    summand1 = tf.einsum('i,i->', ones - targets, LAMBDA * max_1)

    return summand0 + summand1

def train(total_loss, global_step):
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    DECAY_STEPS,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    return apply_gradient_op
