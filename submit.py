import os

cmd = "ipython scripts/sampling/simple_video_sample.py -- --input_path testdata --num_steps 48 --decoding_t 7 --version svd_xt --seed {seed} --output_folder ../../expr/vid/{seed}"

for seed in range(1, 10):
    os.system(cmd.format(seed=seed))
