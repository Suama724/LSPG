import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

data_path = os.path.join(config.ARTIFACTS_DIR, 'latent_samples', 'sampled_points_2026_02_08_092422.npy')

data = np.load(data_path)

point = data[0]
point_pred = np.array([-1.145, 1.458])


distance = np.linalg.norm(point - point_pred)

print(distance)

'''
-1.17, 1.48
-1.14, 1.46

gen: 30
population_size 500
one turn avg 200sec
'''