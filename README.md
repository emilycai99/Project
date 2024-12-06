# Directory description

This repository consists of three directories and a file:

* paper_code: contains the PyTorch source code provided by authors;
* tf_version: contains my TensorFlow implementations;
* tests: contains the test files to check the correctness of TensorFlow implementations.
* Part_I_report.pdf: contains my Part-I report.

# Usage

The get_args.py specifies the arguments.

```bash
usage: train_hnn_tf.py [-h] [--input_dim INPUT_DIM] [--num_samples NUM_SAMPLES] [--len_sample LEN_SAMPLE] [--dist_name DIST_NAME]
                       [--save_dir SAVE_DIR] [--load_dir LOAD_DIR] [--should_load] [--load_file_name LOAD_FILE_NAME]
                       [--total_steps TOTAL_STEPS] [--hidden_dim HIDDEN_DIM] [--num_layers NUM_LAYERS] [--learn_rate LEARN_RATE]
                       [--batch_size BATCH_SIZE] [--batch_size_test BATCH_SIZE_TEST] [--nonlinearity NONLINEARITY]
                       [--test_fraction TEST_FRACTION] [--step_size STEP_SIZE] [--print_every PRINT_EVERY] [--verbose]
                       [--grad_type GRAD_TYPE] [--seed SEED] [--num_hmc_samples NUM_HMC_SAMPLES]
                       [--num_burnin_samples NUM_BURNIN_SAMPLES] [--epsilon EPSILON] [--num_cool_down NUM_COOL_DOWN]
                       [--hnn_threshold HNN_THRESHOLD] [--lf_threshold LF_THRESHOLD] [--gpu_id GPU_ID] [--num_pos NUM_POS] [-f FFF]
```



The following gives an example of training Hamiltonian Neural Network (HNN) for sampling from a 3D Rosenbrock density. The setting is 40 training samples, each with 100 units of end time and a step size of 0.025, the total training steps are 100,000.

```bash
python train_hnn_tf.py --dist_name nD_Rosenbrock --input_dim 6 --num_samples 40 --step_size 0.025 --len_sample 100 --num_layers 3 --total_steps 100000 --verbose --print_every 500 --num_hmc_samples 125000 --gpu_id 3
```

The command below conducts the HMC sampling from the 3D Rosenbrock density, where 125,000 samples are drawn.

```bash
python hnn_nuts_online_tf.py  --dist_name nD_Rosenbrock --input_dim 6 --num_samples 40 --step_size 0.025 --len_sample 100 --num_layers 3 --total_steps 100000 --verbose --print_every 500 --num_hmc_samples 125000 --gpu_id 3
```

To run the tests, one can use:
```bash
pytest
```