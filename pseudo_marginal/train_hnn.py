import os
import sys
import tensorflow as tf
import keras
import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
from nn_models import MLP, CNN_MLP, Info_MLP, Info_CNN_MLP
from hnn import HNN
from data import get_dataset_loader
from utils import L2_loss, log_start, log_stop
from get_args import get_args
from functions import Hamiltonian_func_debug
from grad import calculate_grad

def l1_penalty(args, model):
  if 'cnn' in args.nn_model_name:
    return tf.constant(0.0)
  elif 'info' in args.nn_model_name:
      d = args.target_dim + args.aux_dim
  else:
      d = 2 * (args.target_dim + args.aux_dim)
  for weight in model.trainable_weights:
    if weight.shape[0] == d:
        l1_norm = tf.norm(weight, ord=1)
        # print(weight.name, weight.shape, l1_norm)
  return l1_norm * args.penalty_strength

def train(args, save_path):
  # set random seed
  tf.random.set_seed(args.seed)

  assert args.aux_dim == args.T * args.N, 'aux_dim, T, N do not match'
  args.input_dim = 2 * (args.target_dim + args.aux_dim)

  # initialize Hamiltonian function
  if args.grad_flag:
    cal = calculate_grad(args)
    func = cal.grad_total
  else:
    ham = Hamiltonian_func_debug(args)
    func = ham.get_func()
  
  # Model building
  if args.nn_model_name == 'mlp':
    nn_model = MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                  num_layers=args.num_layers)
  elif args.nn_model_name == 'cnn':
    nn_model = CNN_MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                  num_layers=args.num_layers)
  elif args.nn_model_name == 'info':
    nn_model = Info_MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                  num_layers=args.num_layers)
  elif args.nn_model_name == 'infocnn':
    nn_model = Info_CNN_MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                  num_layers=args.num_layers)
  else:
    raise NotImplementedError
    
  model = HNN(args, differentiable_model=nn_model, grad_type=args.grad_type)
  model(tf.random.normal([args.batch_size, args.input_dim]))
  print(model.summary())

  optim = keras.optimizers.Adam(learning_rate=args.learn_rate, weight_decay=1e-4)

  if args.retrain:
    model.load_weights(save_path+'/best.ckpt')
    optim.lr.assign(args.retrain_lr)

  @tf.function
  def train_step(x, true):
    model.training = True
    with tf.GradientTape() as tape:
      dxdt_hat = model.time_derivative(x)
      l1_loss = l1_penalty(args, model)
      loss = L2_loss(true, dxdt_hat) + l1_loss 
    grads = tape.gradient(loss, model.trainable_weights)
    optim.apply_gradients(zip(grads, model.trainable_weights))
    return loss, grads, l1_loss

  train_dataset, test_dataset = get_dataset_loader(args=args, func=func, seed=args.seed)

  best_test_loss = tf.constant(float('inf'), dtype=tf.float32)
  best_model = None
  best_step = None
  patience = 0
  learning_rate = args.learn_rate if not args.retrain else args.retrain_lr

  # vanilla train loop
  print('Training HNN begins...')
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):

    # training iterate over the train_dataset, different from before
    train_loss = tf.constant(0.0, dtype=tf.float32)
    train_count = 0
    for coords, dcoords in train_dataset:
      with tf.device('/gpu:0'):
        coords_gpu = tf.Variable(coords)
        dcoords_gpu = tf.Variable(dcoords)
      if step == 0:
        pred = model.time_derivative(coords_gpu)
        l1_loss = l1_penalty(args, model)
        loss = L2_loss(coords_gpu, pred) + l1_loss
      else:
        loss, grads, l1_loss = train_step(coords_gpu, dcoords_gpu)
        # print(grads)
      train_loss += loss * coords_gpu.shape[0]
      train_count += coords_gpu.shape[0]
    train_loss = train_loss / train_count

    # run test data
    model.training = False
    test_loss = tf.constant(0.0, dtype=tf.float32)
    test_count = 0
    for test_coords, test_dcoords in test_dataset:
      with tf.device('/gpu:0'):
        test_coords_gpu = tf.Variable(test_coords)
        test_dcoords_gpu_true = tf.Variable(test_dcoords)
      test_dcoords_pred = model.time_derivative(test_coords_gpu)
      test_loss += L2_loss(test_dcoords_gpu_true, test_dcoords_pred) * test_coords_gpu.shape[0]
      test_count += test_coords_gpu.shape[0]
    test_loss = test_loss / test_count

    # logging
    stats['train_loss'].append(train_loss.numpy())
    stats['test_loss'].append(test_loss.numpy())
    if args.verbose and step % args.print_every == 0:
      # print("step {}, sample {}, train_loss {:.2f}, test_loss {:.2f}".format(step, train_count, train_loss.numpy(), test_loss.numpy()))
      print("step {}, sample {}, learn rate {}, train_loss {:.2f}, test_loss {:.2f}, sparsity {:.2f}".format(step, train_count, learning_rate,
                                                                                                             train_loss.numpy() - l1_loss.numpy(),
                                                                                                             test_loss.numpy(), l1_loss.numpy()))
    
    # adjust learning rate
    ## step > 0 because at iteration 0, I want to see what is the initial loss
    if step > 0:
      if best_test_loss > test_loss:
        best_test_loss = test_loss
        best_model = model
        best_step = step
        best_model.save_weights(save_path + '/best.ckpt')
        patience = 0
      elif patience > 3:
        model.load_weights(save_path + '/best.ckpt')
        patience = 0
        learning_rate = learning_rate * args.decay_rate
        optim.lr.assign(learning_rate)
        if learning_rate < 1e-6:
          break
        print("reduce learning rate to {:.2e}, test loss: {:.2f}".format(learning_rate, test_loss))
      else:
        patience += 1

  print('Best test loss {:.2f} at step {}'.format(best_test_loss.numpy(), best_step))

  return model, best_model, stats

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(str(args.gpu_id))
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    result_path = '{}/results/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}_lr{}_ps{}_{}'.format(args.save_dir, args.dist_name,
                                                                           args.T, args.n, args.p, args.N,
                                                                           args.num_samples, args.len_sample, 
                                                                           args.step_size, args.learn_rate, args.penalty_strength,
                                                                           args.nn_model_name)
    if not os.path.exists(result_path):
      os.makedirs(result_path)

    # save
    save_path = '{}/ckp/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}_lr{}_ps{}_{}'.format(args.save_dir, args.dist_name,
                                                                           args.T, args.n, args.p, args.N,
                                                                           args.num_samples, args.len_sample, 
                                                                           args.step_size, args.learn_rate, args.penalty_strength,
                                                                           args.nn_model_name)
    os.makedirs(save_path) if not os.path.exists(save_path) else None

    log_start(result_path+'/log.txt')
    model, best_model, stats = train(args, save_path)
    log_stop()

    model.save_weights(save_path + '/t{}.ckpt'.format(args.total_steps))
    