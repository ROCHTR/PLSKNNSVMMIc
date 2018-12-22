# PLSKNNSVMMIc

Klasifikasi Data Microarray Colon Tumor menggunakan Partial Least Square Regression dan K-Nearest Neighbor - Support Vector Machines

Step By Step Preprocessing
 1. Membagi data menjadi data test dan data train menggunakan K-Fold dengan K = 5. 
 2. Mereduksi dimensi menggunakan Feature Extraction Partial Least Square.

Step By Step Klasifikasi
 1. Data yang belum terdeﬁnisi kelasnya (datatest) akan diberi label kelas menggunakan metode KNN.Metode ini menghitung jarak antara data dengan semua data train menggunakan euclidean distance. 
 2. Kemudian jarak hasil proses sebelumnya akan diurutkan dari yang terkecil ke yang terbesar. 
 3. Jika sejumlah k tetangga terdekat mempunyai label kelas yang sama maka data tersebut diberi label sesuai label k tetangga terdekat. 
 4. Jika tidak maka data train sejumlah k tetangga terdekat akan diproses menggunakan metode SVM dengan euclidean distance sebagai kernel. 
 5. Metode SVM kemudian akan menentukan data tersebut masuk kedalam kelas mana. 
 6. Kemudian data tersebut diberi label sesuai dengan kelas hasil dari klasiﬁkasi metode SVM. 
