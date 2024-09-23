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
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.jet(np.linspace(0, 1, euler.shape[0]))
#scatter = ax.scatter((euler[:, 0] - 1.5) % np.pi, euler[:, 1] + 1.5, (euler[:, 2] - 1.5) % np.pi, color=colors, alpha=0.3, marker='o') # pose 500
#scatter = ax.scatter((euler[:, 0] + 3) % np.pi, (euler[:, 1]) % np.pi, (euler[:, 2] + 1) % np.pi, color=colors, alpha=0.3, marker='o') 
#scatter = ax.scatter((euler[:, 0]+1.5) % np.pi, (euler[:, 1]+2) % np.pi, (euler[:, 2]+2) % np.pi, color=colors, alpha=0.3, marker='o') 
#scatter = ax.scatter((euler[:, 0]+1)%np.pi, euler[:, 1]+1, (euler[:, 2]-1)%np.pi, color=colors, alpha=0.3, marker='o') 

scatter = ax.scatter(euler[:, 0], euler[:, 1], euler[:, 2], color=colors, alpha=0.3, marker='o') 

cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Point Index')
ax.set_title('3D Visualization of SO(3) Rotations')
ax.set_xlabel('Q1')
ax.set_ylabel('Q2')
ax.set_zlabel('Q3')
ax.set_xlim([0, np.pi])
ax.set_ylim([0, np.pi])
ax.set_zlim([0, np.pi])

plt.show()


