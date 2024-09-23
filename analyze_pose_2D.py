import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

with open('hump.pkl', 'rb') as file:
    data = pickle.load(file)
    print("Data loaded from .pkl file:")
    rotations = data[0]


euler = R.from_matrix(rotations).as_euler('XYZ', degrees=False) 

fig = plt.figure()
ax = fig.add_subplot(111)
colors = plt.cm.jet(np.linspace(0, 0.65, euler.shape[0]))
# scatter = ax.scatter((euler[:, 0]+0.3)%np.pi, euler[:, 1]+1.5, color=colors, alpha=0.5)
# scatter = ax.scatter((euler[:, 0]+3)%np.pi, euler[:, 1], color=colors, alpha=0.5)
# scatter = ax.scatter((euler[:, 0] - 1.5) % np.pi, euler[:, 1] + 1.5, color=colors, alpha=0.5)# pose_cell2_lr1e4.pkl
scatter = ax.scatter((euler[:, 0]+1.5)%np.pi, (euler[:, 1]+1.5), color=colors, alpha=0.5)
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
ax.set_title('Visualization of SO(3) Rotations')
ax.set_xlabel('Azimuth')
ax.set_ylabel('Elevation')
ax.set_xlim([-0.5, np.pi+0.5])
ax.set_ylim([-0.5, np.pi+0.5])
plt.show()


