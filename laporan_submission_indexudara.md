# Laporan Proyek _Machine Learning_ - Paramita Citra Indah Mulia

## _Project Overview_

**Latar Belakang:**

Ibu Kota Jakarta sudah lama menghadapi tantangan besar dalam mengatasi polusi udara. Polusi udara di Jakarta tidak hanya mempengaruhi sistem pernapasan, tetapi juga dapat mempengaruhi jantung dan pembuluh darah, meningkatkan risiko serangan jantung, stroke, dan penyakit kardiovaskular lainnya. Anak-anak lebih rentan terhadap efek polusi udara ini karena paparan dapat menghambat perkembangan paru-paru mereka. Selain itu, polusi udara berkontribusi pada perubahan iklim lokal, kerusakan tanaman, dan juga pencemaran air. Hal ini mendorong adanya kebutuhan untuk memahami kualitas udara  agar pemerintah dan masyarakat dapat mengambil tindakan pencegahan maupun mitigasi untuk menjaga kesehatan. Kebutuhan ini dapat dioptimalisasi dengan menggunakan machine learning untuk melakukan klasifikasi kualitas udara di Jakarta.



## _Business Understanding_
### **_Problem Statements:_**
Berdasarkan pada latar belakang di atas, permasalahan pada proyek ini adalah sebagai berikut:
- Bagaimana cara melakukan pemodelan dengan _machine learning_ untuk klasifikasi kualitas udara di Jakarta?
- Apa metode terbaik untuk klasifikasi kualitas udara berdasarkan data yang tersedia?
    
### **_Goals:_**
Tujuan dari proyek ini adalah sebagai berikut:
- Mengembangkan model yang dapat mengklasifikasikan kualitas udara di Jakarta berdasarkan parameter yang diukur.
- Membandingkan dua metode machine learning untuk menentukan metode terbaik dalam klasifikasi kualitas udara.
    
### **_Solution Statements:_**
- Klasifikasi menggunakan Algoritma dua algoritma machine learning yaitu _Random Forest_ dan _KNeighbors (KNN)_
- Model dengan akurasi tertinggi dari data latih adalah model terbaik yang akan digunakan untuk klasifikasi. 



## _Data Understanding_
Data yang digunakan adalah data indeks standar pencemaran udara tahun 2020-2021 yang didapatkan dari **Jakarta Open Data** (https://data.jakarta.go.id/). Data ini terdiri dari 3655 baris (hari) dengan 11 variabel yaitu:

**Data Numerikal:**
```markdown
1. tanggal : Tanggal pengukuran kualitas udara
3. pm10 : Partikulat salah satu parameter yang diukur
4. pm25 : Partikulat salah satu parameter yang diukur
5. so2 : Sulfida (dalam bentuk SO2) salah satu parameter yang diukur
6. co : Karbon Monoksida salah satu parameter yand diukur
7. o3 : Ozon salah satu parameter yang diukur
8. no2 : NItrogen dioksida salah satu parameter yang diukur
9. max : Nilai ukur paling tinggi dari seluruh parameter yang diukur dalam waktu yang sama
```

**Data Kategorikal:**
```markdown
2. stasiun : Lokasi pengukuran di stasiun
10. critical : Parameter yang hasil pengukurannya paling tinggi
11. categori : Kategori hasil perhitungan indeks standar pencemaran udara
```
**EDA**
1. Distribusi kelas:
   ![gambar](https://github.com/yocimm/air_pollution_classification/blob/main/gambar/jml_per_kat.png?raw=true)

   >Terdapat 5 kategori yaitu sedang, baik, tidak sehat, sangat tidak sehat, dan tidak ada data. Kategori 'tidak ada data' akan dihapus, dan karena jumlah data dengan kategori sangat tidak sehat hanya 3 (sangat sedikit), maka pada klasifikasi ini hanya akan       digunakan tiga kategori saja yaitu sedang, baik, dan tidak sehat.

3. Pesebaran kelas di 5 stasiun:
   ![gambar](https://github.com/yocimm/air_pollution_classification/blob/main/gambar/5_stasiun.png?raw=true)

   >Gambar di atas adalah visualisasi dari 5 data berdasarkan stasiun terbanyak. Stasiun Lubang Buaya merupakan daerah    dengan indeks udara tidak sehat terbanyak dan Bunderan HI adalah yang paling sedikit. Sementara itu, seluruh stasiun memiliki kategori sedang yang hampir sama dengan terbanyak adalah stasiun Jagakarsa

5. Jumlah parameter critical:
   ![gambar](https://github.com/yocimm/air_pollution_classification/blob/main/gambar/titik_critical.png?raw=true)
   >Parameter dengan hasil pengukuran tertinggi terbanyak adalah PM25. Menurut Kementerian Lingkungan Hidup dan Kehutanan, parameter PM25 merupakan parameter pencemar udara paling berpengaruh terhadap kesehatan manusia.



## _Data Preparation_
1. _Data Cleaning_

    - Menghapus data dengan kategori SANGAT TIDAK SEHAT dan TIDAK ADA DATA
    - Menyeleksi data dengan nilai pm25 non-null karena merupakan parameter pencemar udara paling berpengaruh terhadap kesehatan. Jumlah data awal 3655 menjadi 1749 data,
    - Melakukan imputasi dengan mean pada missing value fitur 'pm10', 'so2', 'co', 'o3', 'no2', 'pm25'.
    - Menghapus fitur tanggal untuk melakukan pengecekan data duplikat dan didapatkan 0 duplikasi data.

2. Membagi data sesuai label agar memiliki distribusi yang tidak jauh berbeda. Hasil _cleaning_ didapati data 1328 SEDANG, 149 BAIK, 272 TIDAK SEHAT. Oleh karena itu, digunakan 200 SEDANG, 149 BAIK, dan 200 TIDAK SEHAT.

3. Mengubah data kategorik yaitu stasiun, critical, dan categori menjadi numerik dengan _LabelEncoder_.
   - 'categori': {'BAIK': 0, 'SEDANG': 1, 'TIDAK SEHAT': 2},
   - 'critical': {'O3': 0, 'PM10': 1, 'PM25': 2, 'SO2': 3},
   - 'stasiun': {'DKI1 (Bunderan HI)': 0,
                'DKI2 (Kelapa Gading)': 1,
                'DKI3 (Jagakarsa)': 2,
                'DKI4 (Lubang Buaya)': 3,
                'DKI5 (Kebon Jeruk) Jakarta Barat': 4}

5. Memisahkan kolom fitur dengan kolom target yang disimpan secara berturut-turut pada variabel X dan y.

6. Melakukan normalisasi pada data dengan _StandarScaler_.

7. Membagi data training dan testing dengan jumlah data testing sebanyak 20% dari keseluruhan data.

## _Modeling_
Pemodelan pada proyek ini menggunakan 2 algoritma machine learning yaitu _Random Forest Classifier_ dan _K-Nearest Neighbor Classifier_.

  - **_Random Forest_** 
    _Random Forest_ adalah algoritma berbasis _ensemble_ yang menggabungkan beberapa pohon keputusan untuk menghasilkan prediksi. _Random Forest_ mempertimbangkan fitur 'stasiun', 'pm10', 'so2', 'co', 'o3', 'no2', 'max', 'pm25', dan 'critical', kemudian mencoba menemukan pola terbaik untuk memprediksi 'categori' berdasarkan kombinasi fitur-fitur tersebut. Pemodelan ini menggunakan _random forest_ dengan jumlah estimator sebanyak 100 (100 pohon keputusan yang berbeda dari dataset pelatihan). Dengan jumlah tersebut, _random forest_ dapat mengurangi _varians_ yang mungkin ada dalam satu pohon tunggal, sehingga meningkatkan stabilitas dan akurasi model. 
    
  - **_KNN_** dengan jumlah _neighbors_ yang optimal adalah sebanyak 3.
    _KNN_ adalah algoritma berbasis _instance_ yang bekerja dengan membandingkan _instance_ baru dengan _instance_ dalam dataset pelatihan. _KNN_ mempertimbangkan 'k' titik data pelatihan terdekat untuk membuat prediksi, nilai _n_neighbors_ pada pelatihan ini yaitu 3. _KNN_ akan mempertimbangkan fitur 'stasiun', 'pm10', 'so2', 'co', 'o3', 'no2', 'max', 'pm25', dan 'critical' kemudian mencari 3 tetangga terdekat dalam data pelatihan untuk memprediksi 'categori' dari data baru.


## Evaluation
Pada pelatihan ini _random forest_ memiliki nilai akurasi sebesar 0,98.
Pada pelatihan ini KNN memiliki nilai akurasi sebesar 0,90.
Dari hasil akurasi kedua model tersebut, Random Forest adalah model terbaik untuk klasifikasi pada kasus indeks pencemaran udara ini.

Selain penilaian dengan _accurary_score_ digunakan pula metrik evaluasi yaitu _Confusion matrix_ dan _Classification report_.

1. _Confusion Matrix_

Dari data test dilakukan prediksi pada kedua model kemudian dibandingkan dengan nilai aktual dan disajikan dengan metrik berikut:
**_Confusion Matrix Random Forest_:**
![gambar](https://github.com/yocimm/air_pollution_classification/blob/main/gambar/confusion-rf.png?raw=true)

> Dengan _Random Forest_ didapati 2 kesalahan seharusnya kategori SEDANG, namun dikategorikan BAIK.

**_Confusion Matrix KNN_:**
![gambar](https://github.com/yocimm/air_pollution_classification/blob/main/gambar/confusion-knn.png?raw=true)

> Sementara dengan _KNN_ didapati 3 kesalahan seharusnya kategori SEDANG namun dikategorikan BAIK, 4 kesalahan seharusnya kategori TIDAK SEHAT namun dikategorikan SEDANG, dan 4 kesalahan seharusnya kategori SEDANG namun dikategorikan TIDAK SEHAT.



3. _Classification Report_
        
  **_Random Forest_**
    
  ![kelas](https://github.com/yocimm/air_pollution_classification/blob/main/gambar/classreport-rf.png?raw=true)
 
  - _Precision_: Proporsi dari prediksi positif yang benar-benar positif.
      - Untuk Kelas 0, dari semua prediksi yang diberi label sebagai Kelas 0, 90% dari mereka benar-benar adalah Kelas 0.
      - Untuk Kelas 1, dari semua prediksi yang diberi label sebagai Kelas 1, 100% dari mereka benar-benar adalah Kelas 1.
      - Untuk Kelas 2, dari semua prediksi yang diberi label sebagai Kelas 2, 100% dari mereka benar-benar adalah Kelas 2.

  - _Recall_ (Sensitivitas): Proporsi dari observasi positif yang benar-benar diprediksi dengan benar.
    - Untuk Kelas 0, dari semua observasi yang sebenarnya adalah Kelas 0, 100% dari mereka diprediksi dengan benar oleh model.
    - Untuk Kelas 1, dari semua observasi yang sebenarnya adalah Kelas 1, 96% dari mereka diprediksi dengan benar oleh model.
    - Untuk Kelas 2, dari semua observasi yang sebenarnya adalah Kelas 2, 100% dari mereka diprediksi dengan benar oleh model.
    
  - _F1-Score_: Rata-rata harmonik dari _precision_ dan _recall_. Memberikan kesimbangan antara _precision_ dan _recall_.
    - Untuk Kelas 0, _F1-Score_ adalah 0.95.
    - Untuk Kelas 1, _F1-Score_ adalah 0.98.
    - Untuk Kelas 2, _F1-Score_ adalah 1.00.

  **KNN**

  ![gambar](https://github.com/yocimm/air_pollution_classification/blob/main/gambar/classreport-knn.png?raw=true)
  
  - _Precision_: Proporsi dari prediksi positif yang benar-benar positif.
      - Untuk Kelas 0, dari semua prediksi yang diberi label sebagai Kelas 0, 86% dari mereka benar-benar adalah Kelas 0.
      - Untuk Kelas 1, dari semua prediksi yang diberi label sebagai Kelas 1, 91% dari mereka benar-benar adalah Kelas 1.
      - Untuk Kelas 2, dari semua prediksi yang diberi label sebagai Kelas 2, 90% dari mereka benar-benar adalah Kelas 2.

  - _Recall_ (Sensitivitas): Proporsi dari observasi positif yang benar-benar diprediksi dengan benar.
    - Untuk Kelas 0, dari semua observasi yang sebenarnya adalah Kelas 0, 100% dari mereka diprediksi dengan benar oleh model.
    - Untuk Kelas 1, dari semua observasi yang sebenarnya adalah Kelas 1, 86% dari mereka diprediksi dengan benar oleh model.
    - Untuk Kelas 2, dari semua observasi yang sebenarnya adalah Kelas 2, 90% dari mereka diprediksi dengan benar oleh model.
    
  - _F1-Score_: Rata-rata harmonik dari _precision_ dan _recall_. Memberikan kesimbangan antara _precision_ dan _recall_.
    - Untuk Kelas 0, _F1-Score_ adalah 0.93.
    - Untuk Kelas 1, _F1-Score_ adalah 0.89.
    - Untuk Kelas 2, _F1-Score_ adalah 0.90.

**Berdasarkan hasil evaluasi yang telah dilakukan, _Random Forest_ memiliki performa lebih baik daripada _KNeighbors_ dalam melakukan klasifikasi indeks pencemaran udara di Jakarta.**
