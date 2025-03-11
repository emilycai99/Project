import os, sys
import tensorflow as tf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from pseudo_marginal.utils import to_pickle, from_pickle, numerical_grad, integrator
from pseudo_marginal.grad import numerical_grad_debug

def get_trajectory_tf(args, func, t_span=None, timescale=None, y0=None, **kwargs):
    '''
    Description:
        get trajectory for HNN training with auto differentiation
    Args:
        args: from parser
        func: Hamiltonian function
        t_span: a list with two elements [start, end]
        timescale: step size
        y0: initial value
    Return:
        dic1: a list of coords value
            len(dic1) = 2 * (args.target_dim + args.aux_dim), each element is of 1 x (n_steps+1)
        ddic1: a list of numerical gradients 
            len(dic1) = 2 * (args.target_dim + args.aux_dim), each element is of 1 x (n_steps+1)
    '''
    if t_span is None:
        t_span=[0, args.len_sample]
    if timescale is None:
        timescale=args.step_size
    n_steps = int((t_span[1] - t_span[0]) / timescale)

    # initialization
    if y0 is None:
        y0 = tf.random.normal(shape=[2 * (args.target_dim + args.aux_dim)])
    
    # integrator_out.shape = [2 * (args.target_dim + args.aux_dim), n_step + 1]
    integrator_out = integrator(y0, func, numerical_grad, timescale, n_steps, args.target_dim, args.aux_dim)

    dic1 = tf.split(integrator_out, 2 * (args.target_dim + args.aux_dim))
    dydt = [numerical_grad(coords=integrator_out[:, ii], func=func, target_dim=args.target_dim, aux_dim=args.aux_dim)
            for ii in range(integrator_out.shape[1])]
    dydt = tf.stack(dydt, axis=1)
    ddic1 = tf.split(dydt, 2 * (args.target_dim + args.aux_dim))
    # dic1: should be (position, momentum) state
    # dydt: numerical gradient
    # dic1: a list, len(dic1) = 2 * (args.target_dim + args.aux_dim), each element is of 1 x n_steps
    # ddic1: a list, len(ddic1) = 2 * (args.target_dim + args.aux_dim), each element is of 1 x n_steps
    return dic1, ddic1

def get_trajectory_tf_debug(args, func, t_span=None, timescale=None, y0=None, **kwargs):
    '''
    Description:
        get trajectory for HNN training with manual gradients
    Args:
        args: from parser
        func: gradient function
        t_span: a list with two elements [start, end]
        timescale: step size
        y0: initial value
    Return:
        dic1: a list of coords value
            len(dic1) = 2 * (args.target_dim + args.aux_dim), each element is of 1 x (n_steps+1)
        ddic1: a list of numerical gradients 
            len(dic1) = 2 * (args.target_dim + args.aux_dim), each element is of 1 x (n_steps+1)
    '''
    if t_span is None:
        t_span=[0, args.len_sample]
    if timescale is None:
        timescale=args.step_size
    n_steps = int((t_span[1] - t_span[0]) / timescale)

    # initialization
    if y0 is None:
        y0 = tf.random.normal(shape=[2 * (args.target_dim + args.aux_dim)])
    
    # integrator_out.shape = [2 * (args.target_dim + args.aux_dim), n_step + 1]
    integrator_out = integrator(y0, func, numerical_grad_debug, timescale, n_steps, args.target_dim, args.aux_dim)

    dic1 = tf.split(integrator_out, 2 * (args.target_dim + args.aux_dim))
    dydt = [numerical_grad_debug(coords=integrator_out[:, ii], grad_func=func, target_dim=args.target_dim, aux_dim=args.aux_dim)
            for ii in range(integrator_out.shape[1])]
    dydt = tf.stack(dydt, axis=1)
    ddic1 = tf.split(dydt, 2 * (args.target_dim + args.aux_dim))
    # dic1: should be (position, momentum) state
    # dydt: numerical gradient
    # dic1: a list, len(dic1) = 2 * (args.target_dim + args.aux_dim), each element is of 1 x n_steps
    # ddic1: a list, len(ddic1) = 2 * (args.target_dim + args.aux_dim), each element is of 1 x n_steps
    return dic1, ddic1

def get_dataset_tf(args, func, seed=0, samples=None, y_init=None, **kwargs):
    '''
    Description:
        create the dataset for HNN training
    Args:
        args: from parser
        func: hamiltonian function
        seed: random seed
        samples: number of samples
        y_init: initialization for coords
    Return:
        data: a dictionary with keys: '['coords', 'test_coords', 'dcoords', 'test_dcoords']'
    '''
    if samples is None:
        samples = args.num_samples
    
    data = {'meta': locals()}
    # randomly sample inputs
    tf.random.set_seed(seed) 
    xs, dxs = [], []
        
    count1 = 0
    if y_init is None:
        y_init = tf.random.normal(shape=[2*(args.target_dim+args.aux_dim)], dtype=tf.float32)

    print('Generating HMC samples for HNN training')

    folder = '{}/data/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}'.format(args.save_dir, args.dist_name, 
                                                                    args.T, args.n, args.p, args.N,
                                                                    args.num_samples, args.len_sample, args.step_size)
    
    if not os.path.exists(folder+'/train'):
        os.makedirs(folder+'/train')
    if not os.path.exists(folder+'/validation'):
        os.makedirs(folder+'/validation')
    
    ## if args.grad_flag then using the manual gradients; otherwise, auto differentiation
    if args.grad_flag:
        get_trajectory = get_trajectory_tf_debug
    else:
        get_trajectory = get_trajectory_tf

    for s in range(samples):
        print('Sample number ' + str(s+1) + ' of ' + str(samples))
        dic1, ddic1 = get_trajectory(args=args, func=func, y0=y_init, **kwargs)
        # the adding element is of shape step x args.input_dim
        dic1_tmp = tf.transpose(tf.concat(dic1, axis=0))
        xs.append(dic1_tmp)
        ddic1_tmp = tf.transpose(tf.concat(ddic1, axis=0))
        dxs.append(ddic1_tmp)

        count1 = count1 + 1
        # not reuse theta and u from the last step, because it will easily get to extreme values
        y_init = tf.random.normal(shape=[2*(args.target_dim+args.aux_dim)], dtype=tf.float32)
        
        save_obj = tf.stack([dic1_tmp, ddic1_tmp], axis=-1)
        if tf.reduce_any(tf.math.is_nan(save_obj)):
            print('Sample {} contains nan'.format(s+1))
        to_pickle(save_obj, os.path.join(folder+'/train', 'sample{}.pkl'.format(s)))

        
    data['coords'] = tf.concat(xs, axis=0)
    data['dcoords'] = tf.squeeze(tf.concat(dxs, axis=0))

    test_xs = []
    test_dxs = []
    test_samples = int(samples * args.test_fraction)
    for s in range(test_samples):
        print('Sample number (test) ' + str(s+1) + ' of ' + str(test_samples))
        dic1, ddic1 = get_trajectory(args=args, func=func, y0=y_init, **kwargs)
        # the adding element is of shape step x args.input_dim
        test_dic1_tmp = tf.transpose(tf.concat(dic1, axis=0))
        test_xs.append(test_dic1_tmp)
        test_ddic1_tmp = tf.transpose(tf.concat(ddic1, axis=0))
        test_dxs.append(test_ddic1_tmp)

        count1 = count1 + 1
        y_init = tf.random.normal(shape=[2*(args.target_dim+args.aux_dim)], dtype=tf.float32)
        
        save_obj = tf.stack([test_dic1_tmp, test_ddic1_tmp], axis=-1)
        if tf.reduce_any(tf.math.is_nan(save_obj)):
            print('Sample (test) {} contains nan'.format(s+1))
        to_pickle(save_obj, os.path.join(folder+'/validation', 'sample{}.pkl'.format(s)))
    
    data['test_coords'] = tf.concat(test_xs, axis=0)
    data['test_dcoords'] = tf.squeeze(tf.concat(test_dxs, axis=0))

    # return a dictionary with keys: '['coords', 'test_coords', 'dcoords', 'test_dcoords']'
    return data

def load_tensor_slices(filename):
    return from_pickle(filename.numpy())

def load_tensor(x):
    return x[:, 0], x[:, 1]

def get_dataset_loader(args, func, seed=0):
    '''
    Description:
        prepare dataset loader for training
    Args:
        args: from parser
        func: if args.grad_flag, then the func is the gradient function; otherwise, the Hamiltonian
        seed: random seed for reproduction
    Return:
        iterable training set and testing set
    '''
    data_folder = '{}/data/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}'.format(args.save_dir, args.dist_name, 
                                                                      args.T, args.n, args.p, args.N,
                                                                      args.num_samples, args.len_sample, args.step_size)
    if (not args.should_load) or (not os.path.exists(data_folder)):
        print("The path {} does not exists. Begin to generate data...".format(data_folder))
        get_dataset_tf(args=args, func=func, seed=seed)
    
    print('Begin to load data ...')
    # First load the training data
    train_folder = os.path.join(data_folder, 'train')
    filenames = [os.path.join(train_folder, file) for file in os.listdir(train_folder)]
    train_set0 = tf.data.Dataset.from_tensor_slices(filenames)
    train_set1 = train_set0.map(lambda filename: tf.py_function(load_tensor_slices, [filename], [tf.float32]),
                                num_parallel_calls=tf.data.AUTOTUNE)
    train_set2 = train_set1.interleave(lambda x: tf.data.Dataset.from_tensor_slices(x).map(load_tensor),
                                        num_parallel_calls=tf.data.AUTOTUNE)
    train_set3 = train_set2.filter(lambda x, y: tf.logical_and(tf.logical_and(tf.logical_not(tf.reduce_any(tf.math.is_nan(x))), 
                                                                            tf.logical_not(tf.reduce_any(tf.math.is_inf(x)))),
                                                                tf.logical_and(tf.logical_not(tf.reduce_any(tf.math.is_nan(y))), 
                                                                            tf.logical_not(tf.reduce_any(tf.math.is_inf(y))))))
    train_set = train_set3.shuffle(args.shuffle_buffer_size).batch(args.batch_size)

    # Then load the testing data
    test_folder = os.path.join(data_folder, 'validation')
    filenames = [os.path.join(test_folder, file) for file in os.listdir(test_folder)]
    test_set0 = tf.data.Dataset.from_tensor_slices(filenames)
    test_set1 = test_set0.map(lambda filename: tf.py_function(load_tensor_slices, [filename], [tf.float32]),
                                num_parallel_calls=tf.data.AUTOTUNE)
    test_set2 = test_set1.interleave(lambda x: tf.data.Dataset.from_tensor_slices(x).map(load_tensor),
                                        num_parallel_calls=tf.data.AUTOTUNE)
    test_set3 = test_set2.filter(lambda x, y: tf.logical_and(tf.logical_and(tf.logical_not(tf.reduce_any(tf.math.is_nan(x))), 
                                                                            tf.logical_not(tf.reduce_any(tf.math.is_inf(x)))),
                                                                tf.logical_and(tf.logical_not(tf.reduce_any(tf.math.is_nan(y))), 
                                                                            tf.logical_not(tf.reduce_any(tf.math.is_inf(y))))))
    test_set = test_set3.shuffle(args.shuffle_buffer_size).batch(args.batch_size_test)
    return train_set, test_set