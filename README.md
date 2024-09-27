ODT

使用方法

  0.可以调用make_data.py将文件夹中的.tif文件处理为cryodrgn可处理的.mrcs文件(此外还上传了一个onion_1.mrcs,一个洋葱细胞的小数据集)

  1.新建环境，install cryodrgn

  2.在cryodrgn的环境下，在cryodrgn/commands路径下用python(不是cryodrgn abinit_homo)执行 abinit_homo.py，填写输入参数后执行

补充
  
  为适应odt物理模型更改的代码在cryodrgn/lattice.py

fsc使用方法

  0.需要下载eman2或者其他的配准算法先配准两个重建数据
  
  python bin/e2proc3d.py --align=rotate_translate_3d_tree:verbose=1 --alignref="reconstruction_1.mrc" --verbose=9 "reconstruction_2.mrc" "reconstruction_2_aligned.mrc"

  1.调用fsc.py（文件内对应位置处修改路径）
