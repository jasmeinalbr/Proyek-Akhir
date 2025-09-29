| Web app | Tautan web app yang digunakan untuk mengakses model serving. Contoh: [nama-model](https://model-resiko-kredit.herokuapp.com/v1/models/model-resiko-kredit/metadata)|

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
| Performa model | Model mencapai Accuracy = 85.0%, AUC = 90.2%, Precision = 72.1%, dan Recall = 62.9% pada data evaluasi (6,610 contoh). Hasil ini menunjukkan model cukup baik dalam membedakan kategori pendapatan (>50K dan <=50K), meskipun masih ada ruang peningkatan terutama pada recall. |
| Opsi deployment | Model disiapkan untuk *serving* menggunakan TensorFlow Serving + Docker, kemudian di-*deploy* ke Railway. |
| Web app | *(belum diisi, tambahkan link endpoint Railway misalnya: `https://adult-income-ml.railway.app/v1/models/adult_income/metadata`)* |
| Monitoring | Monitoring dilakukan menggunakan Prometheus untuk mengumpulkan metrik model, dengan rencana integrasi Grafana sebagai visualisasi performa dan kesehatan model. |

