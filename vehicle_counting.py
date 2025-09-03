import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\USER\.vscode\car\video.mp4")  # 讀取影片
car=0
text = 'cars:'

# 新增：取得影片尺寸與FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 新增：設定 VideoWriter
out = cv2.VideoWriter(
    r"C:\Users\USER\.vscode\car\output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

#car_cascade = cv2.CascadeClassifier("C:\\Users\\tinwe\\.vscode\\cars.xml")  # 載入車輛分類器(github下載的
# backSub = cv2.bgsegm.createBackgroundSubtractorMOG() #創建背景分離器
backSub = cv2.createBackgroundSubtractorMOG2()       #創建背景分離器

while True:
    ret, frame = cap.read()  # 讀取每一幀影像
    if ret:
        f1 = frame[360:720,0:1280]
        gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)  # 轉換成灰階影像
        blurred = cv2.GaussianBlur(gray, (51, 51), 5)     # 模糊化去除雜訊 (img,(指定區域單位),模糊程度)
        Mask = backSub.apply(blurred)                   #背景提取
        dilat = cv2.dilate(Mask, np.ones((15, 15)))   #影像膨脹 第一個參數為二值化的影像， 第二個參數為使用的捲積 kernel，第三個參數為迭代次數(預設為1)
        line = cv2.line(f1, (25, 200), (1200, 200), (0, 0, 255), 3) #中間的紅線(img, 起始座標, 末點座標, 顏色, 粗細)
        
        cv2.rectangle(frame, (0, 360), (1280, 720), (123, 10, 255), 8) #(選擇的圖片,第一個點,第二個點,線條的顏色,線的粗細)
        cv2.putText(frame, text, (425, 100), cv2.FONT_HERSHEY_SIMPLEX,3, (123, 10, 255), 10, cv2.LINE_8) #(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
        cv2.putText(frame, str(car), (650, 110), cv2.FONT_HERSHEY_SIMPLEX,3, (123, 10, 255), 10, cv2.LINE_8)

        #canny = cv2.Canny(Mask, 30, 150)
        ret, output = cv2.threshold(dilat, 127, 255, cv2.THRESH_BINARY) #二值化 將小於閾值的灰度值設為0，其他值設為最大灰度值。>127 =255, <127 =0
        if ret == 127:
            contours,hierarchy = cv2.findContours(dilat, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #得到輪廓訊息
            for cnt in contours: #取第一條輪廓
                M = cv2.moments(cnt) #取得物體質心
                area = cv2.contourArea(cnt) #白色區域面積
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX=0
                    cY=0
                #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2) #(圖,輪廓本身,繪製所有輪廓,顏色,線粗細)
                # print(area)

                if 23000 > area > 3000: #車子面積大於1000會把車子圈起來
                    cv2.rectangle(f1, (cX-100, cY-100), (cX+100, cY+100), (0, 255, 0), 2) #(選擇的圖片,第一個點,第二個點,線條的顏色,線的粗細)
                    cv2.circle(f1, (cX, cY), 2,(255, 0, 0),5)                            #質心的圓點
                    #print('(' + str(cX) +', ' + str(cY) + ')')
                    
                    if 210 >= cY >= 200: #當車子質心的Y座標過紅線 車子數會+1
                        car = car+1
                        # print(car)   
                    
        #cars = car_cascade.detectMultiScale(gray, 1.1, 3)  # 偵測車輛
        
        #for (x, y, w, h) in cars:
            #cnt = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 畫出偵測框
        
        # 新增：儲存辨識結果到影片
        out.write(frame)

        output= cv2.resize(output, (0, 0), fx=0.5, fy=0.5)
        frame= cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        f1 = cv2.resize(f1, (0, 0), fx=0.5, fy=0.5)

        # cv2.imshow("Video", output)  
        # cv2.imshow("Video1", f1)
        # cv2.imshow("Video2", frame)

        if cv2.waitKey(50) == ord('q'):  # 按下q鍵退出
            break
    else:
        break

cap.release()
out.release()  # 釋放 VideoWriter
cv2.destroyAllWindows() 

           









