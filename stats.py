import csv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from pandas import DataFrame

path = 'test/test-dataset.csv'
path2 = 'stars-dataset.csv'

def get_data(path):
    """Read a CSV star dataset from The HYG Database,
    convert it to a pandas dataframe and return
    two lists: distances and velocities; each position
    corresponding to one star.

    >>> print get_data('test/test-dataset')
    (['134.2282', '257.7320'],
    [4.090012224920605e-05, 1.1765823388101658e-05])
    """
    distances, velocities = [], []
    data = DataFrame.from_csv(path)
    for n, row in data.iterrows():
        dist = row['dist']
        vel = np.linalg.norm([row['vx'],
                              row['vy'], row['vz']])
        # When the distance is 100000 it is unknown
        if dist != 0 and vel != 0 and dist != 100000:
            distances.append(dist*10**(-6))
            # From parsec/year to km/s
            velocities.append(vel*978462)
    return distances, velocities

distances, velocities = get_data(path2)

print '\n'*2
print "Stats for distances:"
print "Number of elements " + str(len(distances))
print "Maximum " + str(max(distances))
print "Minimum " + str(min(distances))
print "Mean " + str(np.mean(distances))
print "Stdev " + str(np.std(distances))

print '\n'*2
print "Stats for velocities:"
print "Number of elements " + str(len(velocities))
print "Maximum " + str(max(velocities))
print "Minimum " + str(min(velocities))
print "Mean " + str(np.mean(velocities))
print "Stdev " + str(np.std(velocities))




correl = np.corrcoef(distances, velocities)[0,1]
print 'Correlation ', correl

fit = np.polyfit(distances, velocities, 1)
fit_fn = np.poly1d(fit)
plt.plot(distances, velocities, 'yo', distances, fit_fn(distances), '--k')
plt.axis([0, 1000, 0, 0.0004])
print fit
#plt.plot(distances, velocities, 'ro')

plt.show()
