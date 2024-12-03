import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Veriyi yükleme
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Veriyi kontrol etme: Her örnek 42 öğe olmalıdır (21 el işareti * 2 koordinat: x ve y)
valid_data = []
valid_labels = []

for i in range(len(data)):
    if len(data[i]) == 42:  # Her veri örneği 42 öğe (21 nokta * 2 koordinat) içermelidir
        valid_data.append(data[i])
        valid_labels.append(labels[i])

# Veriyi numpy dizisine dönüştürme
data = np.asarray(valid_data)
labels = np.asarray(valid_labels)

# Eğitim ve test verilerine ayırma
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Modeli oluşturma
model = RandomForestClassifier()

# Modeli eğitme
model.fit(x_train, y_train)

# Test verisi üzerinde tahmin yapma
y_predict = model.predict(x_test)

# Başarıyı ölçme
score = accuracy_score(y_predict, y_test)

# Sonucu yazdırma
print('{}% of samples were classified correctly !'.format(score * 100))

# Modeli kaydetme
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
