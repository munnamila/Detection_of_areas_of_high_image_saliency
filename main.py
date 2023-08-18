import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_high_saliency(image):
    # input: 画像データ
    # output: 顕著性マップ、高顕著性を表示した画像
    
    img = image.copy()
    
    # 画像サイズ
    height, width = img.shape[0], img.shape[1]
    
    # saliency mapの生成
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    
    # 閾値を設定
    threshold_value = 128
    
    # 二値化
    ret, saliencyMap_binary_map = cv2.threshold(saliencyMap, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 輪郭を検出
    contours, _ = cv2.findContours(saliencyMap_binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # テキストを描画
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = height//300
    font_color = (0, 0, 255)  # テキストの色 (BGR形式:赤)
    thickness = height//200  # テキストの太さ
    
    list_saliency_max_value = []# 顕著性保存用
    
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness)  # 緑色の枠を描画
        
        list_saliency_max_value.append(np.max(saliencyMap[y:y+h, x:x+w]))# 顕著性の最大値を保存する
        
    sorted_list = sorted(list_saliency_max_value, reverse=True)
    rank_list = [sorted_list.index(i) for i in list_saliency_max_value]    
        
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # テキストを描画
        out_img = cv2.putText(img, str(rank_list[index]), (x, y), font, font_scale, font_color, thickness) 
        
    return saliencyMap, out_img