import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score
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
model_rfc = RandomForestClassifier(random_state=42)
model_knn = KNeighborsClassifier()
model_svm = SVC(random_state=42)

# Modeli eğitme
model_rfc.fit(x_train, y_train)
model_knn.fit(x_train, y_train)
model_svm.fit(x_train, y_train)

# Test verisi üzerinde tahmin yapma
y_predict_rfc = model_rfc.predict(x_test)
y_predict_knn = model_knn.predict(x_test)
y_predict_svm = model_svm.predict(x_test)

# Başarıyı ölçme
score_rfc = accuracy_score(y_predict_rfc, y_test)
score_knn = accuracy_score(y_predict_knn, y_test)
score_svm = accuracy_score(y_predict_svm, y_test)

# Diğer metrikleri hesaplama
f1_rfc = f1_score(y_predict_rfc, y_test, average='weighted')
precision_rfc = precision_score(y_predict_rfc, y_test, average='weighted')
recall_rfc = recall_score(y_predict_rfc, y_test, average='weighted')

f1_knn = f1_score(y_predict_knn, y_test, average='weighted')
precision_knn = precision_score(y_predict_knn, y_test, average='weighted')
recall_knn = recall_score(y_predict_knn, y_test, average='weighted')

f1_svm = f1_score(y_predict_svm, y_test, average='weighted')
precision_svm = precision_score(y_predict_svm, y_test, average='weighted')
recall_svm = recall_score(y_predict_svm, y_test, average='weighted')

print('Random Forest F1 Score: {}'.format(f1_rfc))
print('Random Forest Precision: {}'.format(precision_rfc))
print('Random Forest Recall: {}'.format(recall_rfc))

print('KNN F1 Score: {}'.format(f1_knn))
print('KNN Precision: {}'.format(precision_knn))
print('KNN Recall: {}'.format(recall_knn))

print('SVM F1 Score: {}'.format(f1_svm))
print('SVM Precision: {}'.format(precision_svm))
print('SVM Recall: {}'.format(recall_svm))


print('Random Forest Accuracy: {}%'.format(score_rfc * 100))
print('KNN Accuracy: {}%'.format(score_knn * 100))
print('SVM Accuracy: {}%'.format(score_svm * 100))

f = open('models.p', 'wb')
pickle.dump({'model_rfc': model_rfc, 'model_knn': model_knn, 'model_svm': model_svm}, f)
f.close()
