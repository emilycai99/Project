# Directory description

This repository comprises four main directories and two PDF files:

* **paper_code**: contains the PyTorch source code provided by authors (**Part I**);
* **tf_version**: contains my TensorFlow implementations (**Part I**);
* **pseudo_marginal**: contains my adaptations of the efficient NUTS method in Dhulipala et al. (2023) to pseudo-marginal HMC in Alenlov et al. (2021) (**Part II**);
* **tests**: contains the test files to check the correctness of TensorFlow implementations (**Parts I and II**);
* **Part_I_report****.pdf**: presents my Part-I report.
* **Part_II_report****.pdf**: presents my Part-II report.

# Usage

## Part I

### Arguments

The tf_version/get_args.py specifies the arguments available for Part I implementations.

```bash
usage: tf_version/train_hnn_tf.py [-h] [--input_dim INPUT_DIM] [--num_samples NUM_SAMPLES] [--len_sample LEN_SAMPLE] [--dist_name DIST_NAME]
                       [--save_dir SAVE_DIR] [--load_dir LOAD_DIR] [--should_load] [--load_file_name LOAD_FILE_NAME]
                       [--total_steps TOTAL_STEPS] [--hidden_dim HIDDEN_DIM] [--num_layers NUM_LAYERS] [--learn_rate LEARN_RATE]
                       [--batch_size BATCH_SIZE] [--batch_size_test BATCH_SIZE_TEST] [--nonlinearity NONLINEARITY]
                       [--test_fraction TEST_FRACTION] [--step_size STEP_SIZE] [--print_every PRINT_EVERY] [--verbose]
                       [--grad_type GRAD_TYPE] [--seed SEED] [--num_hmc_samples NUM_HMC_SAMPLES]
                       [--num_burnin_samples NUM_BURNIN_SAMPLES] [--epsilon EPSILON] [--num_cool_down NUM_COOL_DOWN]
                       [--hnn_threshold HNN_THRESHOLD] [--lf_threshold LF_THRESHOLD] [--gpu_id GPU_ID] [--num_pos NUM_POS] [-f FFF]
```

### HNN training

The following gives an example of training Hamiltonian Neural Network (HNN) for sampling from a 3D Rosenbrock density. The setting is 40 training samples, each with 100 units of end time and a step size of 0.025. The total training steps are 100,000.

```bash
python tf_version/train_hnn_tf.py --dist_name nD_Rosenbrock --input_dim 6 --num_samples 40 --step_size 0.025 --len_sample 100 --num_layers 3 --total_steps 100000 --verbose --print_every 500 --num_hmc_samples 125000 --gpu_id 3
```

### HMC with efficient NUTS and HNNs

The command below conducts the HMC sampling from the 3D Rosenbrock density, where 125,000 samples are drawn.

```bash
python tf_version/hnn_nuts_online_tf.py  --dist_name nD_Rosenbrock --input_dim 6 --num_samples 40 --step_size 0.025 --len_sample 100 --num_layers 3 --total_steps 100000 --verbose --print_every 500 --num_hmc_samples 125000 --gpu_id 3
```

### Tests

To run the tests, one can use:

```bash
pytest
```

## Part II

### Arguments

The pseudo_marginal/get_args.py specifies the arguments available for Part II implementations.

```
usage: pseudo_marginal/train_hnn.py [-h] [--dist_name DIST_NAME] [--data_pth DATA_PTH] [--p P] [--T T] [--N N]
                    [--n N] [--input_dim INPUT_DIM] [--target_dim TARGET_DIM] [--aux_dim AUX_DIM]
                    [--num_samples NUM_SAMPLES] [--len_sample LEN_SAMPLE] [--step_size STEP_SIZE]
                    [--test_fraction TEST_FRACTION] [--save_dir SAVE_DIR] [--load_dir LOAD_DIR]
                    [--should_load] [--load_file_name LOAD_FILE_NAME] [--nn_out_dim NN_OUT_DIM]
                    [--hidden_dim HIDDEN_DIM] [--num_layers NUM_LAYERS] [--learn_rate LEARN_RATE]
                    [--batch_size BATCH_SIZE] [--batch_size_test BATCH_SIZE_TEST]
                    [--nonlinearity NONLINEARITY] [--grad_type GRAD_TYPE] [--total_steps TOTAL_STEPS]
                    [--print_every PRINT_EVERY] [--verbose] [--seed SEED] [--gpu_id GPU_ID]
                    [--shuffle_buffer_size SHUFFLE_BUFFER_SIZE] [--penalty_strength PENALTY_STRENGTH]
                    [--nn_model_name NN_MODEL_NAME] [--decay_rate DECAY_RATE] [--retrain]
                    [--retrain_lr RETRAIN_LR] [--num_hmc_samples NUM_HMC_SAMPLES]
                    [--num_burnin_samples NUM_BURNIN_SAMPLES] [--epsilon EPSILON]
                    [--num_cool_down NUM_COOL_DOWN] [--hnn_threshold HNN_THRESHOLD]
                    [--lf_threshold LF_THRESHOLD] [--adapt_iter ADAPT_ITER] [--delta DELTA]
                    [--grad_flag] [--grad_mass_flag] [--rho_var RHO_VAR] [-f FFF]
```

### HNN training

This subsection gives examples of training different neural networks for sampling from the generalized linear mixed model (GLMM). The setting is 400 training samples, each with two units of end time and a step size of 0.002. The maximum training steps are 1,000 and the initial learning rate is 0.0001.

* To choose different network architectures, one can use the `nn_model_name` argument. Setting it to 'mlp', 'cnn', or 'infocnn' correspond to the multilayer perceptrons, convolution-based architecture and its variant that focuses on learning the gradients of target density only.

  ```
  python pseudo_marginal/train_hnn.py --num_samples 400 --step_size 0.002 --grad_flag --len_sample 2.0 --num_layers 3 --total_steps 1000 --verbose --print_every 1 --gpu_id 0 --seed 0 --batch_size 512 --batch_size_test 512 --learn_rate 1e-4 --decay_rate 0.5 --nn_model_name mlp --should_load
  ```
* To use the $l_1$-regularization, one can set the argument `penalty_strength` to a positive value.

  ```
  python pseudo_marginal/train_hnn.py --num_samples 400 --step_size 0.002 --grad_flag --len_sample 2.0 --num_layers 3 --total_steps 300 --verbose --print_every 1 --gpu_id 0 --seed 0 --batch_size 512 --batch_size_test 512 --learn_rate 1e-4 --decay_rate 0.5 --nn_model_name mlp --should_load --penalty_strength 1.0
  ```

### Pseudo-marginal HMC

This subsection gives examples of pseudo-marginal HMC sampling from the GLMM. There are several sampling schemes available to choose and their commands are given below.

* With efficient NUTS and HNNs: `pseudo_marginal/hnn_nuts_online.py`

  ```
  python pseudo_marginal/hnn_nuts_online.py --num_samples 400 --step_size 0.002 --epsilon 0.002 --len_sample 2.0 --num_layers 3 --total_steps 1000 --verbose --print_every 1 --gpu_id 0 --seed 0 --batch_size 512 --batch_size_test 512 --learn_rate 1e-4 --decay_rate 0.5 --nn_model_name cnn --should_load --num_hmc_samples 14000 --num_burnin_samples 5000

  ```
* With efficient NUTS and numerical gradients: `pseudo_marginal/hnn_nuts_online_num.py`

  ```
  python pseudo_marginal/hnn_nuts_online_num.py --grad_flag --grad_mass_flag --num_samples 400 --step_size 0.002 --epsilon 0.002 --len_sample 2.0 --num_layers 3 --total_steps 1000 --verbose --print_every 1 --gpu_id 0 --seed 0 --batch_size 512 --batch_size_test 512 --learn_rate 1e-4 --decay_rate 0.5 --nn_model_name cnn --should_load --num_hmc_samples 14000 --num_burnin_samples 5000 --hnn_threshold 1000

  ```

  Note that in this particular setting, there are three options on how to calculate the Hamiltonian and numerical gradients by `grad_flag` and `grad_mass_flag`. Setting `grad_mass_flag=True` uses the non-identity mass matrix and manual calculations. Setting `grad_mass_flag=False, grad_flag=True` uses the identity mass matrix and manual calculations. Setting `grad_flag=False` uses `tfp.distributions` and auto-differentiation.
* With efficient NUTS, numerical gradients, and auto-tuning of step size: `pseudo_marginal/hnn_nuts_online_epsilon.py`;

  ```
  python pseudo_marginal/hnn_nuts_online_epsilon.py --grad_flag --num_samples 400 --step_size 0.002 --epsilon 0.002 --len_sample 2.0 --num_layers 3 --total_steps 1000 --verbose --print_every 1 --gpu_id 0 --seed 0 --batch_size 512 --batch_size_test 512 --learn_rate 1e-4 --decay_rate 0.5 --nn_model_name cnn --should_load --num_hmc_samples 14000 --num_burnin_samples 5000 --hnn_threshold 1000
  ```
* Without efficient NUTS or HNNs: `pseudo_marginal/hnn_nuts_online_num_no_nuts.py`.

  ```
  python pseudo_marginal/hnn_nuts_online_num_no_nuts.py --grad_flag --num_samples 400 --step_size 0.002 --epsilon 0.002 --len_sample 2.0 --num_layers 3 --total_steps 1000 --verbose --print_every 1 --gpu_id 0 --seed 0 --batch_size 512 --batch_size_test 512 --learn_rate 1e-4 --decay_rate 0.5 --nn_model_name cnn --should_load --num_hmc_samples 14000 --num_burnin_samples 5000
  ```

### Tests

To run the tests, one can use:

```bash
pytest
```
