

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [nama dataset](https://www.kaggle.com/) |
| Masalah | Deskripsi masalah yang di angkat |
| Solusi machine learning | Deskripsi solusi machine learning yang akan dibuat |
| Metode pengolahan | Deskripsi metode pengolahan data yang digunakan |
| Arsitektur model | Deskripsi arsitektur model yang diguanakan |
| Metrik evaluasi | Deksripsi metrik yang digunakan untuk mengevaluasi performa model |
| Performa model | Deksripsi performa model yang dibuat |
| Opsi deployment | Deksripsi tentang opsi deployment |
| Web app | Tautan web app yang digunakan untuk mengakses model serving. Contoh: [nama-model](https://model-resiko-kredit.herokuapp.com/v1/models/model-resiko-kredit/metadata)|
| Monitoring | Deksripsi terkait hasil monitoring dari model serving |

# Submission 1: Adult Income Prediction

Nama:  Jasmein Al-baar Putri Rus'an
Username dicoding: jasmeinalbaar  

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Adult Income Prediction Dataset](https://www.kaggle.com/datasets/mosapabdelghany/adult-income-prediction-dataset) |
| Masalah | Bagaimana memprediksi apakah seseorang memiliki penghasilan lebih dari 50K USD per tahun berdasarkan data demografi dan pekerjaan. |
| Solusi machine learning | Membangun model klasifikasi biner menggunakan TensorFlow Extended (TFX) pipeline untuk memprediksi label penghasilan (`<=50K` atau `>50K`). |
| Metode pengolahan | Data diproses menggunakan `Transform` dengan `tensorflow_transform` (tft). <br> - Categorical features di-*encode* menggunakan `tft.compute_and_apply_vocabulary` + one-hot encoding. <br> - Numerical features dinormalisasi dengan `tft.scale_to_0_1`. <br> - Label `income` diubah menjadi biner (0 = <=50K, 1 = >50K). |
| Arsitektur model | Model dibangun menggunakan `tf.keras.Sequential` dengan lapisan: <br> - Input layer: gabungan categorical + numerical features. <br> - Dense layer 1: ReLU, units dituning (64–256). <br> - Dense layer 2: ReLU, units dituning (16–128). <br> - Output layer: Dense(1, sigmoid). <br> Optimizer: Adam, Loss: Binary Crossentropy, Metrik: Binary Accuracy. |
| Metrik evaluasi | Evaluasi menggunakan TFMA dengan metrik: <br> - BinaryAccuracy (dengan threshold) <br> - AUC <br> - Precision <br> - Recall <br> - ExampleCount |
| Performa model | *(belum diisi, tambahkan hasil akurasi/precision/recall/AUC dari evaluasi TFMA)* |
| Opsi deployment | Model disiapkan untuk *serving* menggunakan TensorFlow Serving + Docker, kemudian di-*deploy* ke Railway. |
| Web app | *(belum diisi, tambahkan link endpoint Railway misalnya: `https://adult-income-ml.railway.app/v1/models/adult_income/metadata`)* |
| Monitoring | Logging model dilakukan dengan TensorBoard. <br> Hyperparameter tuning menggunakan `KerasTuner`. <br> Rencana integrasi monitoring ke Grafana untuk observabilitas. |

