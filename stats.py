import csv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def get_data(path):
    """Read a CSV star dataset from The HYG Database and return
    two lists: distances and velocities; each position
    corresponding to one star.

    >>> print get_data('test/test-dataset')
    (['134.2282', '257.7320'],
    [4.090012224920605e-05, 1.1765823388101658e-05])
    """
    distances = []
    velocities = []

    with open(path, 'rb') as csvfile:
        stars = list(csv.reader(csvfile, delimiter=' ', quotechar='|'))[1:]

        for star in stars:
            try:  # Dealing with errors in the dataset.
                star = ', '.join(star).split(',')
                dist = star[9]
                velocity = [float(star[20]), float(star[21]),
                            float(star[22])]
                # If there is a velocity and distance for that star.
                if list(set(velocity))[0] and dist:
                    distances.append(float(dist))
                    # Velocity is in vector form, get magnitude,
                    velocities.append(sqrt(np.dot(velocity, velocity)))
            except: pass
    return distances, velocities


distances, velocities = get_data('stars-dataset.csv')
plt.plot(distances, velocities, 'ro')
plt.show()
