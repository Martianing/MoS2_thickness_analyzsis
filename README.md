# $MoS_{2}$ thickness analyzsis
本程序用来自动识别光学显微镜下二硫化钼厚度的变化。通过读取视频文件，在手动框选样品和衬底位置后，可以自动计算二者之间的RGB、HSV、Lab各色彩空间的差值，从而定性分析样品层数的变化。  
本程序基于Python，需要以下库：  
```pyhton
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from matplotlib.colors import hsv_to_rgb
import cv2
```
# 参考文献  
《_Rapid and Reliable Thickness Identification of Two-Dimensional Nanosheets Using Optical Microscopy_》
