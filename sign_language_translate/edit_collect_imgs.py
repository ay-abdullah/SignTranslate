#Bu kod sayfası hatalı datayı düzeltmek için eklendii class to fizx den düzeltilmesi gerekn vri sınıfını seçin

import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class_to_fix = 20  # Düzeltilmesi gereken sınıf
dataset_size = 100

cap = cv2.VideoCapture(0)

# Düzeltilmesi gereken sınıf için işlem
class_dir = os.path.join(DATA_DIR, str(class_to_fix))
if not os.path.exists(class_dir):
    os.makedirs(class_dir)

print('Collecting data for class {}'.format(class_to_fix))

# Kullanıcının başlamak için 'Q' tuşuna basmasını beklemesi
while True:
    ret, frame = cap.read()
    cv2.putText(frame, 'Press Q to start collecting images', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,165,0), 2,
                cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break

# Yeni veriler toplama
counter = 0
while counter < dataset_size:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(25)

    # Görüntüyü kaydet
    cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
    counter += 1

cap.release()
cv2.destroyAllWindows()
