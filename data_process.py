import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from matplotlib.colors import hsv_to_rgb
import cv2


#------------------------------------------------------------------------------------

def get_materials_areas(video_path, output_txt='material_roi_coordinates.txt', output_image='material_roi_image.png'):
    """
    允许用户在视频的第一帧上选择多个矩形区域，并将这些区域的坐标保存到txt文件中，同时生成一张图片记录选中的区域。
    
    参数：
    video_path: 视频文件路径
    output_txt: 保存坐标的txt文件路径
    output_image: 保存选中区域图片的文件路径
    """
    # 初始化视频捕捉对象
    cap = cv2.VideoCapture(video_path)

    # 检查是否成功打开视频文件
    if not cap.isOpened():
        print("无法打开视频文件。")
        return

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频文件的第一帧。")
        cap.release()
        return

    roi_list = []
    frame_copy = frame.copy()
    drawing = False
    roi_start = (0, 0)

    def select_roi(event, x, y, flags, param):
        nonlocal drawing, roi_start, frame_copy, frame
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            roi_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, roi_start, (x, y), (0, 255, 0), 2)
                cv2.imshow('frame', frame_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi_end = (x, y)
            roi_list.append((roi_start, roi_end))
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, roi_start, roi_end, (0, 255, 0), 2)
            cv2.imshow('frame', frame_copy)

    cv2.imshow('frame', frame)
    cv2.setMouseCallback('frame', select_roi)

    print("请用鼠标在视频上框选一个矩形区域，完成后按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    with open(output_txt, 'a') as f:
        for roi in roi_list:
            (x1, y1), (x2, y2) = roi
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            f.write(f"{x1},{y1},{x2},{y2}\n")
            print(f"矩形区域的坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # 保存带有矩形区域的图片
    for roi in roi_list:
        (x1, y1), (x2, y2) = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output_image, frame)
    print(f"保存选中区域图片到 {output_image}")

    # 释放视频捕捉对象
    cap.release()


#------------------------------------------------------------------------------------

def process_video_with_rois(video_path, roi_file='material_roi_coordinates.txt'):
    # 初始化视频捕捉对象
    cap = cv2.VideoCapture(video_path)

    # 检查是否成功打开视频文件
    if not cap.isOpened():
        print("无法打开视频文件。")
        return None

    # 读取ROI坐标
    rois = []
    with open(roi_file, 'r') as f:
        for line in f:
            x1, y1, x2, y2 = map(int, line.strip().split(','))
            rois.append((x1, y1, x2, y2))

    # 初始化用于存储每个ROI的数据
    roi_data = {i: [] for i in range(len(rois))}

    # 遍历每一帧并处理
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 对当前帧进行中值滤波去噪
        frame_denoised = cv2.medianBlur(frame, 5)

        for i, (x1, y1, x2, y2) in enumerate(rois):
            # 裁剪出选中的矩形区域
            roi_frame = frame_denoised[y1:y2, x1:x2]

            # 将裁剪区域转换为float类型进行计算
            roi_frame_float = roi_frame.astype(np.float64)

            # 计算矩形区域内所有像素的RGB总值和均值
            R_total = np.sum(roi_frame_float[:, :, 2])
            G_total = np.sum(roi_frame_float[:, :, 1])
            B_total = np.sum(roi_frame_float[:, :, 0])
            num_pixels = roi_frame_float.shape[0] * roi_frame_float.shape[1]
            R_mean = R_total / num_pixels
            G_mean = G_total / num_pixels
            B_mean = B_total / num_pixels

            # 转换为HSV色彩空间并计算均值
            roi_hsv = cv2.cvtColor(roi_frame.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float64)
            H_mean = np.mean(roi_hsv[:, :, 0])
            S_mean = np.mean(roi_hsv[:, :, 1])
            V_mean = np.mean(roi_hsv[:, :, 2])

            # 转换为Lab色彩空间并计算均值
            roi_lab = cv2.cvtColor(roi_frame.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float64)
            L_mean = np.mean(roi_lab[:, :, 0])
            a_mean = np.mean(roi_lab[:, :, 1])
            b_mean = np.mean(roi_lab[:, :, 2])

            # 记录当前帧的RGB、HSV、Lab均值
            roi_data[i].append([frame_number, R_total, G_total, B_total, R_mean, G_mean, B_mean,
                                H_mean, S_mean, V_mean, L_mean, a_mean, b_mean])

        frame_number += 1

    # 释放视频捕捉对象
    cap.release()

    # 创建并保存数据表格
    dataframes = {}
    for i, (x1, y1, x2, y2) in enumerate(rois):
        df = pd.DataFrame(roi_data[i], columns=['frame_number', 'R_total', 'G_total', 'B_total', 'R_mean', 'G_mean', 'B_mean',
                                                'H_mean', 'S_mean', 'V_mean', 'L_mean', 'a_mean', 'b_mean'])
        csv_filename = f'material_rgb_hsv_lab_values_roi_{x1}_{y1}_{x2}_{y2}.csv'
        df.to_csv(csv_filename, index=False)
        dataframes[(x1, y1, x2, y2)] = df
        print(f"生成并保存数据表格到 {csv_filename}")

    return dataframes

#------------------------------------------------------------------------------------
def plot_dataframe_color_trends(dataframes):
    for roi, df in dataframes.items():
        x1, y1, x2, y2 = roi
        # 创建图表
        plt.figure(figsize=(15, 18))

        # 绘制RGB总值和均值随时间变化的折线图，并填充颜色
        plt.subplot(3, 1, 1)
        plt.plot(df['frame_number'], df['R_mean'], label='R mean', color='red')
        plt.plot(df['frame_number'], df['G_mean'], label='G mean', color='green')
        plt.plot(df['frame_number'], df['B_mean'], label='B mean', color='blue')
        plt.fill_between(df['frame_number'], df['R_mean'], color='red', alpha=0.3)
        plt.fill_between(df['frame_number'], df['G_mean'], color='green', alpha=0.3)
        plt.fill_between(df['frame_number'], df['B_mean'], color='blue', alpha=0.3)
        plt.title('RGB Mean Values Trend')
        plt.xlabel('Frame Number')
        plt.ylabel('RGB Mean Value')
        plt.legend()
        plt.grid(True)

        # 绘制HSV均值随时间变化的折线图，并填充颜色
        plt.subplot(3, 1, 2)
        plt.plot(df['frame_number'], df['H_mean'], label='H mean', color='orange')
        plt.plot(df['frame_number'], df['S_mean'], label='S mean', color='magenta')
        plt.plot(df['frame_number'], df['V_mean'], label='V mean', color='cyan')
        plt.fill_between(df['frame_number'], df['H_mean'], color='orange', alpha=0.3)
        plt.fill_between(df['frame_number'], df['S_mean'], color='magenta', alpha=0.3)
        plt.fill_between(df['frame_number'], df['V_mean'], color='cyan', alpha=0.3)
        plt.title('HSV Mean Values Trend')
        plt.xlabel('Frame Number')
        plt.ylabel('HSV Mean Value')
        plt.legend()
        plt.grid(True)

        # 绘制Lab均值随时间变化的折线图，并填充颜色
        plt.subplot(3, 1, 3)
        plt.plot(df['frame_number'], df['L_mean'], label='L mean', color='black')
        plt.plot(df['frame_number'], df['a_mean'], label='a mean', color='green')
        plt.plot(df['frame_number'], df['b_mean'], label='b mean', color='blue')
        plt.fill_between(df['frame_number'], df['L_mean'], color='black', alpha=0.3)
        plt.fill_between(df['frame_number'], df['a_mean'], color='green', alpha=0.3)
        plt.fill_between(df['frame_number'], df['b_mean'], color='blue', alpha=0.3)
        plt.title('Lab Mean Values Trend')
        plt.xlabel('Frame Number')
        plt.ylabel('Lab Mean Value')
        plt.legend()
        plt.grid(True)

        # 调整子图之间的间距
        plt.tight_layout()

        # 保存图表为图片文件
        output_image_file = f'material_rgb_hsv_lab_trends_{x1}_{y1}_{x2}_{y2}.png'
        plt.savefig(output_image_file)
        print(f"保存图表到 {output_image_file}")

        # 显示图表
        plt.show()


#------------------------------------------------------------------------------------

def plot_csv_color_trends(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 创建图表
    plt.figure(figsize=(15, 18))

    # 绘制RGB总值和均值随时间变化的折线图，并填充颜色
    plt.subplot(3, 1, 1)
    plt.plot(df['frame_number'], df['R_mean'], label='R mean', color='red')
    plt.plot(df['frame_number'], df['G_mean'], label='G mean', color='green')
    plt.plot(df['frame_number'], df['B_mean'], label='B mean', color='blue')
    plt.fill_between(df['frame_number'], df['R_mean'], color='red', alpha=0.3)
    plt.fill_between(df['frame_number'], df['G_mean'], color='green', alpha=0.3)
    plt.fill_between(df['frame_number'], df['B_mean'], color='blue', alpha=0.3)
    plt.title('RGB Mean Values Trend')
    plt.xlabel('Frame Number')
    plt.ylabel('RGB Mean Value')
    plt.legend()
    plt.grid(True)

    # 绘制HSV均值随时间变化的折线图，并填充颜色
    plt.subplot(3, 1, 2)
    plt.plot(df['frame_number'], df['H_mean'], label='H mean', color='orange')
    plt.plot(df['frame_number'], df['S_mean'], label='S mean', color='magenta')
    plt.plot(df['frame_number'], df['V_mean'], label='V mean', color='cyan')
    plt.fill_between(df['frame_number'], df['H_mean'], color='orange', alpha=0.3)
    plt.fill_between(df['frame_number'], df['S_mean'], color='magenta', alpha=0.3)
    plt.fill_between(df['frame_number'], df['V_mean'], color='cyan', alpha=0.3)
    plt.title('HSV Mean Values Trend')
    plt.xlabel('Frame Number')
    plt.ylabel('HSV Mean Value')
    plt.legend()
    plt.grid(True)

    # 绘制Lab均值随时间变化的折线图，并填充颜色
    plt.subplot(3, 1, 3)
    plt.plot(df['frame_number'], df['L_mean'], label='L mean', color='black')
    plt.plot(df['frame_number'], df['a_mean'], label='a mean', color='green')
    plt.plot(df['frame_number'], df['b_mean'], label='b mean', color='blue')
    plt.fill_between(df['frame_number'], df['L_mean'], color='black', alpha=0.3)
    plt.fill_between(df['frame_number'], df['a_mean'], color='green', alpha=0.3)
    plt.fill_between(df['frame_number'], df['b_mean'], color='blue', alpha=0.3)
    plt.title('Lab Mean Values Trend')
    plt.xlabel('Frame Number')
    plt.ylabel('Lab Mean Value')
    plt.legend()
    plt.grid(True)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图表为图片文件
    output_image_file = csv_file.replace('.csv', '_trends.png')
    plt.savefig(output_image_file)
    print(f"保存图表到 {output_image_file}")

    # 显示图表
    plt.show()

#------------------------------------------------------------------------------------

#下面的函数专门用于识别衬底的数据，依次只能接收一个方框
# 全局变量
roi_selected = False
roi = []

def select_roi(event, x, y, flags, param):
    global roi_selected, roi
    frame = param  # 通过param参数访问frame变量
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        roi.append((x, y))
        roi_selected = True
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, roi[0], roi[1], (0, 255, 0), 2)
        cv2.imshow('frame', frame_copy)

def get_subdstrate_data(video_path):
    global roi_selected, roi

    # 初始化视频捕捉对象
    cap = cv2.VideoCapture(video_path)

    # 检查是否成功打开视频文件
    if not cap.isOpened():
        print("无法打开视频文件。")
        return

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频文件的第一帧。")
        cap.release()
        return

    cv2.imshow('frame', frame)
    roi_selected = False
    roi = []
    cv2.setMouseCallback('frame', select_roi, frame)  # 将frame作为参数传递给回调函数

    print("请用鼠标在视频上框选一个矩形区域，然后按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not roi_selected or len(roi) != 2:
        print("矩形区域未正确选择，程序退出。")
        cap.release()
        return

    x1, y1 = roi[0]
    x2, y2 = roi[1]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # 用于存储每一帧的RGB、HSV、Lab均值
    roi_data = []

    # 遍历每一帧并处理
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 对当前帧进行中值滤波去噪
        frame_denoised = cv2.medianBlur(frame, 5)

        # 裁剪出选中的矩形区域
        roi_frame = frame_denoised[y1:y2, x1:x2]

        # 将裁剪区域转换为float类型进行计算
        roi_frame_float = roi_frame.astype(np.float64)

        # 计算矩形区域内所有像素的RGB总值和均值
        R_total = np.sum(roi_frame_float[:, :, 2])
        G_total = np.sum(roi_frame_float[:, :, 1])
        B_total = np.sum(roi_frame_float[:, :, 0])
        num_pixels = roi_frame_float.shape[0] * roi_frame_float.shape[1]
        R_mean = R_total / num_pixels
        G_mean = G_total / num_pixels
        B_mean = B_total / num_pixels

        # 转换为HSV色彩空间并计算均值
        roi_hsv = cv2.cvtColor(roi_frame.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float64)
        H_mean = np.mean(roi_hsv[:, :, 0])
        S_mean = np.mean(roi_hsv[:, :, 1])
        V_mean = np.mean(roi_hsv[:, :, 2])

        # 转换为Lab色彩空间并计算均值
        roi_lab = cv2.cvtColor(roi_frame.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float64)
        L_mean = np.mean(roi_lab[:, :, 0])
        a_mean = np.mean(roi_lab[:, :, 1])
        b_mean = np.mean(roi_lab[:, :, 2])

        # 记录当前帧的RGB、HSV、Lab均值
        roi_data.append([frame_number, R_total, G_total, B_total, R_mean, G_mean, B_mean,
                         H_mean, S_mean, V_mean, L_mean, a_mean, b_mean])

        frame_number += 1

    # 释放视频捕捉对象
    cap.release()

    # 创建数据表格
    df = pd.DataFrame(roi_data, columns=['frame_number', 'R_total', 'G_total', 'B_total', 'R_mean', 'G_mean', 'B_mean',
                                         'H_mean', 'S_mean', 'V_mean', 'L_mean', 'a_mean', 'b_mean'])

    # 保存为CSV文件
    csv_filename = f'substrate_values_{x1}_{y1}_{x2}_{y2}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"保存数据到 {csv_filename}")

    # 创建图表
    plt.figure(figsize=(15, 18))

    # 绘制RGB均值随时间变化的折线图，并填充颜色
    plt.subplot(3, 1, 1)
    plt.plot(df['frame_number'], df['R_mean'], label='R mean', color='red')
    plt.plot(df['frame_number'], df['G_mean'], label='G mean', color='green')
    plt.plot(df['frame_number'], df['B_mean'], label='B mean', color='blue')
    plt.fill_between(df['frame_number'], df['R_mean'], color='red', alpha=0.3)
    plt.fill_between(df['frame_number'], df['G_mean'], color='green', alpha=0.3)
    plt.fill_between(df['frame_number'], df['B_mean'], color='blue', alpha=0.3)
    plt.title('RGB Mean Values Trend')
    plt.xlabel('Frame Number')
    plt.ylabel('RGB Mean Value')
    plt.legend()
    plt.grid(True)

    # 绘制HSV均值随时间变化的折线图，并填充颜色
    plt.subplot(3, 1, 2)
    plt.plot(df['frame_number'], df['H_mean'], label='H mean', color='orange')
    plt.plot(df['frame_number'], df['S_mean'], label='S mean', color='magenta')
    plt.plot(df['frame_number'], df['V_mean'], label='V mean', color='cyan')
    plt.fill_between(df['frame_number'], df['H_mean'], color='orange', alpha=0.3)
    plt.fill_between(df['frame_number'], df['S_mean'], color='magenta', alpha=0.3)
    plt.fill_between(df['frame_number'], df['V_mean'], color='cyan', alpha=0.3)
    plt.title('HSV Mean Values Trend')
    plt.xlabel('Frame Number')
    plt.ylabel('HSV Mean Value')
    plt.legend()
    plt.grid(True)

    # 绘制Lab均值随时间变化的折线图，并填充颜色
    plt.subplot(3, 1, 3)
    plt.plot(df['frame_number'], df['L_mean'], label='L mean', color='black')
    plt.plot(df['frame_number'], df['a_mean'], label='a mean', color='green')
    plt.plot(df['frame_number'], df['b_mean'], label='b mean', color='blue')
    plt.fill_between(df['frame_number'], df['L_mean'], color='black', alpha=0.3)
    plt.fill_between(df['frame_number'], df['a_mean'], color='green', alpha=0.3)
    plt.fill_between(df['frame_number'], df['b_mean'], color='blue', alpha=0.3)
    plt.title('Lab Mean Values Trend')
    plt.xlabel('Frame Number')
    plt.ylabel('Lab Mean Value')
    plt.legend()
    plt.grid(True)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图表为图片文件
    output_image_file = f'subtrate_rgb_hsv_lab_trends_{x1}_{y1}_{x2}_{y2}.png'
    plt.savefig(output_image_file)
    print(f"保存图表到 {output_image_file}")

    # 显示图表
    plt.show()
    return df

#------------------------------------------------------------------------------------

def calculate_brightness_difference(material_csv, substrate_csv):
    # 读取CSV文件
    material_df = pd.read_csv(material_csv)
    substrate_df = pd.read_csv(substrate_csv)
    
    # 对齐frame_number
    merged_df = pd.merge(material_df, substrate_df, on='frame_number', suffixes=('_material', '_substrate'))
    
    # 计算每一帧中的V均值差异
    v_mean_difference = merged_df['V_mean_material'] - merged_df['V_mean_substrate']

    # 计算RGB色彩空间中的亮度
    material_brightness = 0.299 * merged_df['R_mean_material'] + 0.587 * merged_df['G_mean_material'] + 0.114 * merged_df['B_mean_material']
    substrate_brightness = 0.299 * merged_df['R_mean_substrate'] + 0.587 * merged_df['G_mean_substrate'] + 0.114 * merged_df['B_mean_substrate']
    brightness_difference = material_brightness - substrate_brightness

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'frame_number': merged_df['frame_number'],
        'V_mean_difference': v_mean_difference,
        'RGB_brightness_difference': brightness_difference
    })

    # 保存结果到CSV文件
    # result_csv_filename = 'brightness_difference.csv'
    # result_df.to_csv(result_csv_filename, index=False)
    # print(f"保存亮度差结果到 {result_csv_filename}")

    # 绘制亮度差异的趋势图
    plt.figure(figsize=(15, 10))

    # 绘制V均值差异的折线图，并填充颜色
    plt.subplot(2, 1, 1)
    plt.plot(result_df['frame_number'], result_df['V_mean_difference'], label='V Mean Difference', color='blue')
    plt.fill_between(result_df['frame_number'], result_df['V_mean_difference'], color='blue', alpha=0.3)
    plt.title('V Mean Difference Trend')
    plt.xlabel('Frame Number')
    plt.ylabel('V Mean Difference')
    plt.legend()
    plt.grid(True)

    # 绘制RGB亮度差异的折线图，并填充颜色
    plt.subplot(2, 1, 2)
    plt.plot(result_df['frame_number'], result_df['RGB_brightness_difference'], label='RGB Brightness Difference', color='red')
    plt.fill_between(result_df['frame_number'], result_df['RGB_brightness_difference'], color='red', alpha=0.3)
    plt.title('RGB Brightness Difference Trend')
    plt.xlabel('Frame Number')
    plt.ylabel('RGB Brightness Difference')
    plt.legend()
    plt.grid(True)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图表为图片文件
    output_image_file = 'brightness_difference_trends.png'
    plt.savefig(output_image_file)
    print(f"保存亮度差趋势图到 {output_image_file}")

    # 显示图表
    plt.show()

    return result_df

#------------------------------------------------------------------------------------

def plot_color_difference_trends(material_csv, substrate_csv):
    # 读取CSV文件
    material_df = pd.read_csv(material_csv)
    substrate_df = pd.read_csv(substrate_csv)
    
    # 对齐frame_number
    merged_df = pd.merge(material_df, substrate_df, on='frame_number', suffixes=('_material', '_substrate'))
    
    # 计算RGB颜色分量的差值
    R_difference = merged_df['R_mean_material'] - merged_df['R_mean_substrate']
    G_difference = merged_df['G_mean_material'] - merged_df['G_mean_substrate']
    B_difference = merged_df['B_mean_material'] - merged_df['B_mean_substrate']

    # 计算HSV颜色分量的差值
    H_difference = merged_df['H_mean_material'] - merged_df['H_mean_substrate']
    S_difference = merged_df['S_mean_material'] - merged_df['S_mean_substrate']
    V_difference = merged_df['V_mean_material'] - merged_df['V_mean_substrate']

    # 计算Lab颜色分量的差值
    L_difference = merged_df['L_mean_material'] - merged_df['L_mean_substrate']
    a_difference = merged_df['a_mean_material'] - merged_df['a_mean_substrate']
    b_difference = merged_df['b_mean_material'] - merged_df['b_mean_substrate']

    # 创建图表
    plt.figure(figsize=(15, 18))

    # 绘制RGB颜色分量差值的趋势图
    plt.subplot(3, 1, 1)
    plt.plot(merged_df['frame_number'], R_difference, label='R Difference', color='red')
    plt.plot(merged_df['frame_number'], G_difference, label='G Difference', color='green')
    plt.plot(merged_df['frame_number'], B_difference, label='B Difference', color='blue')
    plt.fill_between(merged_df['frame_number'], R_difference, color='red', alpha=0.3)
    plt.fill_between(merged_df['frame_number'], G_difference, color='green', alpha=0.3)
    plt.fill_between(merged_df['frame_number'], B_difference, color='blue', alpha=0.3)
    plt.title('RGB Color Components Difference')
    plt.xlabel('Frame Number')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)

    # 绘制HSV颜色分量差值的趋势图
    plt.subplot(3, 1, 2)
    plt.plot(merged_df['frame_number'], H_difference, label='H Difference', color='orange')
    plt.plot(merged_df['frame_number'], S_difference, label='S Difference', color='magenta')
    plt.plot(merged_df['frame_number'], V_difference, label='V Difference', color='cyan')
    plt.fill_between(merged_df['frame_number'], H_difference, color='orange', alpha=0.3)
    plt.fill_between(merged_df['frame_number'], S_difference, color='magenta', alpha=0.3)
    plt.fill_between(merged_df['frame_number'], V_difference, color='cyan', alpha=0.3)
    plt.title('HSV Color Components Difference')
    plt.xlabel('Frame Number')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)

    # 绘制Lab颜色分量差值的趋势图
    plt.subplot(3, 1, 3)
    plt.plot(merged_df['frame_number'], L_difference, label='L Difference', color='black')
    plt.plot(merged_df['frame_number'], a_difference, label='a Difference', color='green')
    plt.plot(merged_df['frame_number'], b_difference, label='b Difference', color='blue')
    plt.fill_between(merged_df['frame_number'], L_difference, color='black', alpha=0.3)
    plt.fill_between(merged_df['frame_number'], a_difference, color='green', alpha=0.3)
    plt.fill_between(merged_df['frame_number'], b_difference, color='blue', alpha=0.3)
    plt.title('Lab Color Components Difference')
    plt.xlabel('Frame Number')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图表为图片文件
    output_image_file = 'color_components_difference_trends.png'
    plt.savefig(output_image_file)
    print(f"保存颜色分量差异趋势图到 {output_image_file}")

    # 显示图表
    plt.show()

#------------------------------------------------------------------------------------

from skimage.color import rgb2lab, deltaE_ciede2000

def calculate_color_differences(material_csv, substrate_csv):
    """
    计算material与substrate文件之间的像素数据的欧氏距离和CIEDE2000色差，并绘制趋势图
    """
    # 读取CSV文件
    material_df = pd.read_csv(material_csv)
    substrate_df = pd.read_csv(substrate_csv)
    
    # 打印列名以检查是否存在问题
    print("Material CSV columns:", material_df.columns)
    print("Substrate CSV columns:", substrate_df.columns)
    
    # 确保列名存在
    required_columns = ['frame_number', 'R_mean', 'G_mean', 'B_mean']
    for col in required_columns:
        if col not in material_df.columns or col not in substrate_df.columns:
            raise KeyError(f"列 '{col}' 在CSV文件中不存在。")

    # 对齐frame_number
    merged_df = pd.merge(material_df, substrate_df, on='frame_number', suffixes=('_material', '_substrate'))
    
    # 计算欧氏距离和CIEDE2000色差
    euclidean_distances = []
    ciede2000_differences = []
    
    for i in range(len(merged_df)):
        row1 = merged_df.iloc[i]

        # 计算欧氏距离
        euclidean_distance = np.sqrt((row1['R_mean_material'] - row1['R_mean_substrate'])**2 +
                                     (row1['G_mean_material'] - row1['G_mean_substrate'])**2 +
                                     (row1['B_mean_material'] - row1['B_mean_substrate'])**2)
        euclidean_distances.append(euclidean_distance)

        # 计算CIEDE2000色差
        rgb1 = [row1['R_mean_material'] / 255.0, row1['G_mean_material'] / 255.0, row1['B_mean_material'] / 255.0]
        rgb2 = [row1['R_mean_substrate'] / 255.0, row1['G_mean_substrate'] / 255.0, row1['B_mean_substrate'] / 255.0]
        lab1 = rgb2lab(np.array(rgb1).reshape(1, 1, 3))
        lab2 = rgb2lab(np.array(rgb2).reshape(1, 1, 3))
        ciede2000_difference = deltaE_ciede2000(lab1, lab2)[0][0]
        ciede2000_differences.append(ciede2000_difference)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'frame_number': merged_df['frame_number'],
        'euclidean_distance': euclidean_distances,
        'ciede2000_difference': ciede2000_differences
    })

    # 保存结果到CSV文件
    # result_csv_filename = 'color_differences.csv'
    # result_df.to_csv(result_csv_filename, index=False)
    # print(f"保存颜色差异结果到 {result_csv_filename}")

    # 绘制颜色差异的趋势图
    plt.figure(figsize=(15, 10))

    # 绘制欧氏距离差异的折线图，并填充颜色
    plt.subplot(2, 1, 1)
    plt.plot(result_df['frame_number'], result_df['euclidean_distance'], label='Euclidean Distance', color='blue')
    plt.fill_between(result_df['frame_number'], result_df['euclidean_distance'], color='blue', alpha=0.3)
    plt.title('Euclidean Distance Trend')
    plt.xlabel('Frame Number')
    plt.ylabel('Euclidean Distance')
    plt.legend()
    plt.grid(True)

    # 绘制CIEDE2000色差的折线图，并填充颜色
    plt.subplot(2, 1, 2)
    plt.plot(result_df['frame_number'], result_df['ciede2000_difference'], label='CIEDE2000 Difference', color='red')
    plt.fill_between(result_df['frame_number'], result_df['ciede2000_difference'], color='red', alpha=0.3)
    plt.title('CIEDE2000 Difference Trend')
    plt.xlabel('Frame Number')
    plt.ylabel('CIEDE2000 Difference')
    plt.legend()
    plt.grid(True)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图表为图片文件
    output_image_file = 'color_differences_trends.png'
    plt.savefig(output_image_file)
    print(f"保存颜色差异趋势图到 {output_image_file}")

    # 显示图表
    plt.show()

    return result_df

