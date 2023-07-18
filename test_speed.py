"""Test speed of code for updating & evaluation."""

#!/usr/bin/env python
"""
Evaluate performance of predictor model.

Apply linear MPC with provided predictor model to CityLearn environment
with specified dataset to evaluate predictor performance.
"""

import os
import csv
import time
import numpy as np

from citylearn.citylearn import CityLearnEnv



if __name__ == '__main__':

    dataset_dir = os.path.join('citylearn_challenge_2022_phase_3')   # dataset directory
    schema_path = os.path.join('citylearn','data', dataset_dir, 'schema.json')

    # initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)
    observations = env.reset()

    num_steps = 0
    done = False

    step_time = 0

    # Execute control loop.
    while not done:
        actions = [[val] for val in np.random.uniform(-1,1,len(env.buildings))]
        start = time.time()
        observations, _, done, _ = env.step(actions)
        end = time.time()
        step_time += end - start

        num_steps += 1

    start = time.time()
    metrics = env.evaluate()
    end = time.time()
    eval_time = end - start

    print("Step: %s s" % round(step_time,5))
    print("Eval: %s s" % round(eval_time,5))
