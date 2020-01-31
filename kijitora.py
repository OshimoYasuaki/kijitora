import cv2
import matplotlib.pyplot as plt

cat_cascade = cv2.CascadeClassifier('cascade.xml')

# 画像ファイルの読込
loaded_img = cv2.imread("kiji.jpg")

# グレースケールに変換
gray = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2GRAY)

# 検出の実行 
faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

# 矩形線の色
rectangle_color = (0, 255, 0) #緑色
 
# 検出した場合
if len(faces) > 0:
    for rect in faces:
        cv2.rectangle(loaded_img, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), rectangle_color, thickness=7)

detection = cv2.cvtColor(loaded_img, cv2.COLOR_RGB2BGR) #RGBからBGRに変換
plt.imshow(detection)
