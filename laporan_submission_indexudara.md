# Laporan Proyek Machine Learning - Paramita Citra Indah Mulia

## Project Overview

**Latar Belakang:**

Ibu Kota Jakarta sudah lama menghadapi tantangan besar dalam mengatasi polusi udara. Polusi udara di Jakarta tidak hanya mempengaruhi sistem pernapasan, tetapi juga dapat mempengaruhi jantung dan pembuluh darah, meningkatkan risiko serangan jantung, stroke, dan penyakit kardiovaskular lainnya. Anak-anak lebih rentan terhadap efek polusi udara ini karena paparan dapat menghambat perkembangan paru-paru mereka. Selain itu, polusi udara berkontribusi pada perubahan iklim lokal, kerusakan tanaman, dan juga pencemaran air. Hal ini mendorong adanya kebutuhan untuk memahami kualitas udara  agar pemerintah dan masyarakat dapat mengambil tindakan pencegahan maupun mitigasi untuk menjaga kesehatan. Kebutuhan ini dapat dioptimalisasi dengan menggunakan machine learning untuk melakukan klasifikasi kualitas udara di Jakarta.


## Business Understanding
### **Problem Statements:**
Berdasarkan pada latar belakang di atas, permasalahan pada proyek ini adalah sebagai berikut:
- Bagaimana kita dapat mengklasifikasikan kualitas udara di Jakarta berdasarkan parameter yang diukur?
 - Apa metode terbaik untuk klasifikasi kualitas udara berdasarkan data yang tersedia?
    
### **Goals:**
Tujuan dari proyek ini adalah sebagai berikut:
- Mengembangkan model yang dapat mengklasifikasikan kualitas udara di Jakarta berdasarkan parameter yang diukur.
- Membandingkan dua metode machine learning untuk menentukan metode terbaik dalam klasifikasi kualitas udara.
    
### **Solution Statements:**
- Klasifikasi menggunakan Algoritma dua algoritma machine learning yaitu Random Forest dan KNeighbors
- Model dengan akurasi tertinggi dari data latih adalah model terbaik yang akan digunakan untuk klasifikasi. 


## Data Understanding
Data yang digunakan adalah data indeks standar pencemaran udara tahun 2020-2021 yang didapatkan dari Jakarta Open Data (https://data.jakarta.go.id/). Data ini terdiri dari 3655 baris (hari) dengan 11 variabel yaitu:

**Data Numerikal:**
```markdown
1. tanggal : Tanggal pengukuran kualitas udara
2. stasiun : Lokasi pengukuran di stasiun
3. pm10 : Partikulat salah satu parameter yang diukur
4. pm25 : Partikulat salah satu parameter yang diukur
5. so2 : Sulfida (dalam bentuk SO2) salah satu parameter yang diukur
6. co : Carbon Monoksida salah satu parameter yand diukur
7. o3 : Ozon salah satu parameter yang diukur
8. no2 : NItrogen dioksida salah satu parameter yang diukur
9. max : Nilai ukur paling tinggi dari seluruh parameter yang diukur dalam waktu yang sama
```

**Data Kategorikal:**
```markdown
10. critical : Parameter yang hasil pengukurannya paling tinggi
11. categori : Kategori hasil perhitungan indeks standar pencemaran udara
```
**EDA**


Terdapat pula visualisasi data tentang jumlah data berdasarkan label dan didapatkan insight :
terdapat 5 kategori yaitu sedang, baik, tidak sehat, sangat tidak sehat, dan tidak ada data. Kategori 'tidak ada data' akan dihapus, dan karena jumlah data dengan kategori sangat tidak sehat hanya 3, maka pada klasifikasi ini hanya akan digunakan tiga kategori saja yaitu sedang, baik, dan tidak sehat.



## Data Preparation
Pada tahapan ini dilakukan pendefinisian data berdsarkan 3 kategori yang telah ditetapkan. Kemudian juga dilakukan penghapusan pada kolom tanggal dan stasiun karena kolom tersebut tidak akan digunakan untuk klasifikasi. Setelah itu dilakukan pengecekan jumlah data dan didapatkan data untuk kolom pm25 hanya sebanyak 1749 dari 3655 data. Sedangkan pm25 merupakan salah satu faktor penting dalam klasifikasi kategori pencemaran udara, oleh karena itu data yang akan digunakan adalah data yang memiliki pm25 non-nul. Kemudian dilakukan pengecekan missing value serta duplikasi data dan dilakukan penanganan agar data bersih untuk pemodelan. Setelah data bersih tersisa 1328 data kategori sedang, 149 kategori baik, dan 272 kategori tidak sehat. Melihat jumlahnya, data-data ini dipilih sebanyak 200 data kategori sedang, 149 kategori baik, dan 200 data kategori tidak seha untuk dilakukan pembagian data menjadi train dan test.

## Modeling
Dilakukan pemodelan dengan 2 algoritma yaitu Random Forest dan SVM (Support Vector Machine).

Kedua algoritma mendapati akurasi sebesar 100%, namun pada dasarnya kedua algoritma ini memiliki kelebihan dan kekurangan masing-masing yaitu:

**Random Forest:**

Kelebihan:

* Mengurangi overfitting: Dengan menggunakan banyak pohon, Random Forest cenderung mengurangi overfitting yang mungkin terjadi pada pohon tunggal.
* Pentingnya fitur: Random Forest memberikan skor pentingnya fitur, yang dapat membantu dalam seleksi fitur.
* Bisa digunakan untuk klasifikasi dan regresi: Random Forest fleksibel untuk berbagai jenis tugas.
* Dapat menangani data besar: Dengan kemampuannya untuk membangun banyak pohon, ia dapat menangani dataset dengan jumlah sampel yang besar.
* Dapat menangani fitur kategorikal: Tidak memerlukan one-hot encoding.

Kekurangan:

* Kompleksitas Model: Model Random Forest bisa menjadi kompleks dan memerlukan lebih banyak waktu untuk pelatihan dibandingkan dengan pohon keputusan.
* Tidak sepenuhnya interpretatif: Meskipun lebih interpretatif dibandingkan SVM, Random Forest dengan banyak pohon mungkin sulit untuk diinterpretasi.
* Lambat dalam prediksi: Karena harus melewati banyak pohon saat prediksi, prosesnya bisa lebih lambat dibandingkan dengan model lain.


**Support Vector Machines (SVM):**

Kelebihan:

* Efektif di ruang dimensi tinggi: SVM bekerja dengan baik ketika ada banyak fitur.
* Memori Efisien: SVM menggunakan subset dari data pelatihan dalam fungsi keputusan (disebut vektor pendukung), sehingga lebih hemat memori.
* Fleksibel: Kernel trick memungkinkan SVM untuk menciptakan batas keputusan yang kompleks, bahkan jika data tidak linier.
* Robust terhadap overfitting: Terutama di ruang dimensi tinggi.

Kekurangan:

* Tidak cocok untuk dataset besar: Pelatihan SVM pada dataset yang sangat besar bisa menjadi sangat lambat.
* Sensitif terhadap noise: Sebuah sedikit noise dan pencilan dapat mempengaruhi margin yang dihasilkan oleh SVM.
* Memerlukan pemilihan kernel yang tepat: Pemilihan kernel yang salah dapat menyebabkan model yang tidak sesuai untuk data.
* Hasil tidak langsung dapat diinterpretasi: Tidak seperti pohon keputusan, hasil dari SVM tidak mudah diinterpretasi oleh manusia.

**Karena pada proyek ini hanya menggunakan dataset dengnan ukuran kecil, maka melihat kompleksitasnya, SVM adalah model terbaik.**

## Evaluation
Evaluasi dilakukan menggunakan confussion matrix. Dari confussion matrix memperlihatkan bahwa kedua algoritma mampu dengan sangat baik memprediksi kategori indeks udara di porvinsi jakarta. 19 label 0 (BAIK), 50 label 1 (SEDANG), dan 41 label 2 (TIDAK SEHAT) pada data tes dapat diprediksi semua benar sehingga nilai accuracy = 1, precission = 1, recall = 1, dan f1-score = 1.
