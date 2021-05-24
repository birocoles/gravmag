import matplotlib.pyplot as plt
import numpy as np

def model_boundaries(model, m2km=True):
    '''
    Plot the projection of the model boundaries on plane xy.
    '''
    for prism in model:
        y1, y2, x1, x2 = prism[:4]
        x = np.array([x1, x2, x2, x1, x1])
        y = np.array([y1, y1, y2, y2, y1])
        if m2km is True:
            plt.plot(0.001*y, 0.001*x, 'k--', linewidth=2)
        else:
            plt.plot(y, x, 'k--', linewidth=2)
