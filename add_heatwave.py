"""Stochastic addition of heatwave to loaded temperature data."""

import os
import csv
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

from citylearn.citylearn import CityLearnEnv



if __name__ == '__main__':

    dataset_dir = os.path.join('citylearn_challenge_2022_phase_3')   # dataset directory
    dataset_path = os.path.join('citylearn','data', dataset_dir)

    seed = 0
    np.random.seed(seed)

    # grab weather data
    weather = pd.read_csv(os.path.join(dataset_path,'weather.csv'))

    # plot original summer temperatures
    start_hour = 24*31*5
    n_hours = 24*31*3
    f = plt.figure()
    plt.plot(range(start_hour,start_hour+n_hours),weather['Outdoor Drybulb Temperature [C]'][start_hour:start_hour+n_hours])
    plt.xlim(start_hour,start_hour+n_hours)

    # randomly create heatwave temp multiplier series
    heatwave_amplitude = 0.8 # additional amplitude of heatwave peak - i.e. x fraction hotter than normal
    heatwave_hours = np.random.randint(7*24,10*24) # length of heatwave in hours (between 7 and 10 days - arbitrary choice)
    heatwave_location = np.random.randint(0,n_hours-heatwave_hours) # hour on which the heatwave starts
    heatwave_additions = tukey(heatwave_hours,alpha=96/heatwave_hours) # shape of heatwave bump

    heatwave_multipliers = np.ones(n_hours)
    heatwave_multipliers[heatwave_location:heatwave_location+heatwave_hours] += heatwave_amplitude*heatwave_additions

    g = plt.figure()
    plt.plot(range(start_hour,start_hour+n_hours),heatwave_multipliers)
    plt.xlim(start_hour,start_hour+n_hours)

    # plot adjusted temperatures
    heatwave_temperatures = weather['Outdoor Drybulb Temperature [C]'][start_hour:start_hour+n_hours]*heatwave_multipliers
    h = plt.figure()
    plt.plot(range(start_hour,start_hour+n_hours),heatwave_temperatures)
    plt.xlim(start_hour,start_hour+n_hours)

    plt.show()