import matplotlib.pyplot as plt
import numpy as np
from cryodrgn import fft, mrc
import argparse
import time

Apix = 1.8

def fsc(vol1, vol2):
    vol1, _ = mrc.parse_mrc(vol1)
    vol2, _ = mrc.parse_mrc(vol2)

    D = vol1.shape[0]
    x = np.arange(-D // 2, D // 2)
    x2, x1, x0 = np.meshgrid(x, x, x, indexing="ij")
    coords = np.stack((x0, x1, x2), -1)
    r = (coords**2).sum(-1) ** 0.5

    assert r[D // 2, D // 2, D // 2] == 0.0

    vol1 = fft.fftn_center(vol1)
    vol2 = fft.fftn_center(vol2)

    prev_mask = np.zeros((D, D, D), dtype=bool)
    fsc = [1.0]
    for i in range(1, D // 2):
        mask = r < i
        shell = np.where(mask & np.logical_not(prev_mask))
        v1 = vol1[shell]
        v2 = vol2[shell]
        p = np.vdot(v1, v2) / (np.vdot(v1, v1) * np.vdot(v2, v2)) ** 0.5
        fsc.append(float(p.real))
        prev_mask = mask
    fsc = np.asarray(fsc)
    x = np.arange(D // 2) / D
    res = np.stack((x, fsc), 1)
    return res

def fsc_res(vol1, vol2):
    
    vol1, _ = mrc.parse_mrc(vol1)
    vol2, _ = mrc.parse_mrc(vol2)

    assert isinstance(vol1, np.ndarray)
    assert isinstance(vol2, np.ndarray)

    D = vol1.shape[0]
    x = np.arange(-D // 2, D // 2)
    x2, x1, x0 = np.meshgrid(x, x, x, indexing="ij")
    coords = np.stack((x0, x1, x2), -1)
    r = (coords**2).sum(-1) ** 0.5

    assert r[D // 2, D // 2, D // 2] == 0.0

    vol1 = fft.fftn_center(vol1)
    vol2 = fft.fftn_center(vol2)

    prev_mask = np.zeros((D, D, D), dtype=bool)
    fsc = [1.0]
    for i in range(1, D // 2):
        mask = r < i
        shell = np.where(mask & np.logical_not(prev_mask))
        v1 = vol1[shell]
        v2 = vol2[shell]
        p = np.vdot(v1, v2) / (np.vdot(v1, v1) * np.vdot(v2, v2)) ** 0.5
        fsc.append(float(p.real))
        prev_mask = mask
    fsc = np.asarray(fsc)
    print(fsc)
    x = np.arange(D // 2) / D
    # w = np.where(fsc < 0.5)
    # res05 = x[w[0][0]-1] - (fsc[w[0][0]-1] - 0.5) / ((fsc[w[0][0]-1] - fsc[w[0][0]]) / (x[w[0][0]-1] - x[w[0][0]]))
    w = np.where(fsc < 0.143)
    res0143 = x[w[0][0]-1] - (fsc[w[0][0]-1] - 0.143) / ((fsc[w[0][0]-1] - fsc[w[0][0]]) / (x[w[0][0]-1] - x[w[0][0]]))
    return 1 / res0143 * Apix


import os

path_template_A = '/home/pc/Desktop/cs_reader/particles_and_ctf/analyze_J1343_new_50/kmeans20/blurred_files/vol_{:03d}_aligned_4zt0.mrc'
path_B = '/home/pc/Desktop/cs_reader/particles_and_ctf/analyze_J1343_new_50/kmeans20/blurred_files/blurred_4zt0_1.8.mrc'

# 循环处理路径 A 中的所有文件
for i in range(20):
    path_A = path_template_A.format(i)
    if os.path.exists(path_A):
        res_fsc = fsc(path_A, path_B)

        # 绘制并保存每个文件的 FSC 曲线
        x_axis = res_fsc[:, 0]
        y_axis = res_fsc[:, 1]
        resolution = x_axis / Apix

        plt.plot(resolution, y_axis, label=f'FSC vol_{i:03d}')
        plt.ylim((0, 1))
        plt.title(f'FSC curve for vol_{i:03d}', fontsize=15)
        plt.xlabel('1/resolution ($\mathrm{\AA}^{-1}$)', fontsize=15)
        plt.ylabel('FSC score', fontsize=15)
        plt.xticks([0, 1/18, 1/8, 1/4, 1/2], [0, r'$1/18$', r'$1/8$', r'$1/4$', r'$1/2$'], fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=15)
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

        output_file = f'FSC_curve_vol_{i:03d}_zt0.png'
        plt.savefig(output_file, dpi=600)
        plt.clf()  # 清除当前图，以便下一个文件使用
        print(f'FSC for vol_{i:03d} done, saved as {output_file}')
    else:
        print(f'File {path_A} does not exist, skipping.')