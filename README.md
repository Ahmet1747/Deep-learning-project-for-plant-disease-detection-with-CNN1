# Deep learning project for plant disease detection with CNN1
Görüntülerden bitki yaprağı hastalıklarını otomatik olarak tespit eden, Convolutional Neural Network (CNN) tabanlı derin öğrenme sistemi. Hassas tarımı desteklemek ve erken teşhis sağlamak amacıyla geliştirilmiştir

Kütüphaneleri Ekleme
![image](https://github.com/user-attachments/assets/553b37d0-7d2e-4700-a464-c058ce2c7da4)
Kütüphaneler import ediliyor.Görüntü verileri ile çalıştığımız için tensorflow,matplatlib, pandas ve keras kütüphanelerini kullandık.

Veri Ön işlemesi
Burda bitki hastalıklarının görüntüleri ile görüntü veri seti oluşturmak için kerasın image datasını kullandık.Ve eğitim setimiz hazır hale geldi.
![image](https://github.com/user-attachments/assets/5dd7290b-0c7d-493c-a711-728607f25c4b)

Burda da doğrulama veri setini oluşturduk. Bu veri seti, modelin eğitim süreci sırasında doğrulama amacıyla kullanılır. Modelin doğrulama sırasında ne kadar iyi performans gösterdiğini değerlendirebilmek için bu veri setine ihtiyaç vardır. Doğrulama veri seti oluşturuluyor.
![image](https://github.com/user-attachments/assets/8bb415f2-7c31-4bcb-b4e0-d320b60ea8ad)

Burda training_setindeki verilerin doğru bir şekide yüklenip yüklenmediğine ve yığınlardaki görüntülerin,etiketlerin boyutlarını görmek için kullandık.
![image](https://github.com/user-attachments/assets/0bb49cba-829c-4f9b-a9b6-36ff30d324ac)

Burda TensorFlow Keras API'sından gerekli katmanları ve model türünü içeri aktarıyoruz. Dense tam bağlantılı katmandır.Son veride kullanılır.Conv2D ise evrişim katmanımızdır.MaxPool2D ise evrişim katmnalarından sonra özellik haritalarının boyutunu küçültmek için kullanılır.Flatten katmanı iki boyutlu verimizi tek boyutlu veriye dönüştürmek için kullanılır.Dropout modelde aşırı öğrenmeyi engellemk için kullanılır.Sequintial modeli ise katmanları sıralı bir şekilde kulanmak için kullanılır.
![image](https://github.com/user-attachments/assets/ef3b585d-2b0e-47f3-96c2-fb02e11e7f70)
Bir model nesnesi oluşturur. Bu nesne üzerine farklı katmanlar eklenerek modelin yapısı inşa edilebilir.
![image](https://github.com/user-attachments/assets/755f48b6-171b-4309-80da-a608aa795147)
filters=32: Bu, katman tarafından kullanılacak olan filtre sayısını belirtir. 32 farklı filtre ile görüntü üzerinde evrişim işlemi yapılacaktır.
    • kernel_size=3: Filtre boyutunu belirtir. Burada 3x3 boyutunda bir filtre kullanılır, yani her filtre 3x3'lük bir bölge üzerinde işlem yapacaktır.
    • padding='same': Görüntünün boyutunun değişmemesini sağlar. Yani, giriş görüntüsünün boyutuna bağlı olarak, gerekli padding (doldurma) işlemi yapılır. Bu, çıkış boyutunun giriş boyutuyla aynı olacağı anlamına gelir.
    • activation='relu': ReLU (Rectified Linear Unit) aktivasyon fonksiyonu kullanılır. Bu, negatif değerleri sıfırlar ve pozitif değerleri olduğu gibi bırakır, bu da doğrusal olmayan bir dönüşüm sağlar.
    • input_shape=[128, 128, 3]: Bu parametre, modelin ilk katmanına verilecek olan giriş görüntülerinin boyutlarını belirtir. 128x128 boyutunda ve 3 renk kanalına (RGB) sahip görüntüler beklenmektedir. 
İkinci satır  padding='same' parametresi belirtilmediği için, varsayılan olarak padding='valid' kullanılır. Bu da çıkış boyutunun giriş boyutundan daha küçük olacağı anlamına gelir. 
Üçüncü satır ise 2x2 havuzlama penceresi kullanarak çıkış boyutunu 2'ye indirger ve hesaplama yükünü azaltır. 
![image](https://github.com/user-attachments/assets/759a4f24-3e01-4d21-9795-f8e8c1d27e3c)
Filters=64,128,256,512: Bu katmanlarda tarafından kullanılacak olan filtre sayısını belirtir. Filtre sayısını her seferinde arttırarak  görüntü üzerinde evrişim işlemi yapılmıştır.
![image](https://github.com/user-attachments/assets/12e58444-67ab-4d99-9132-2592aa27aa11)
Dropout(0.25) katmanı, modelin %25'lik bir oranla eğitim sırasında nöronları devre dışı bırakmasını sağlar. Bu, aşırı uyumun önlenmesine yardımcı olur ve modelin daha genel bir çözüm öğrenmesini destekler. 
![image](https://github.com/user-attachments/assets/8287b610-131d-48e5-8d97-c885e683df7d)
Flatten() katmanı, çok boyutlu veriyi tek boyutlu bir vektöre dönüştürür.Böylelikle modelimizi düzleştirmiş oluyoruz.
![image](https://github.com/user-attachments/assets/c05a0bda-29cb-4121-b75c-ac3d11c4f56b)
Dense(units=1500, activation='relu') katmanı, modelin öğrenme kapasitesini artıran, 1500 nörondan oluşan tam bağlantılı bir katmandır. Yani gizli katmanımızdır.Bu katman, modelin daha ileri düzeyde özellikleri öğrenmesini sağlar. ReLU aktivasyonu, negatif değerleri sıfırlayarak modelin doğrusal olmayan ilişkileri öğrenmesine yardımcı olur. Bu tür gizli katmanlar , derin öğrenme ağlarında genellikle modelin genelleme yeteneğini artırmak ve daha karmaşık desenleri öğrenmesini sağlamak amacıyla kullanılır. 
![image](https://github.com/user-attachments/assets/08468748-69ff-4c11-9160-44cf1be91031)
Bu kodda da %40 ile nöronları devre dışı bırakıyoruz. Aşırı öğrenmeyi engellemek için.
![image](https://github.com/user-attachments/assets/8a04fb02-67b4-4be4-91d5-f0ffe19d3ec4)
Softmax  fonksiyonu: Softmax, genellikle çok sınıflı sınıflandırma (multi-class classification) görevlerinde kullanılır. Bu fonksiyon, her sınıfın tahmin edilen olasılığını hesaplar ve modelin hangi sınıfı seçmesi gerektiği konusunda karar verir. 
 Bu, modelin her bir sınıf için tahmin ettiği olasılığı hesaplamasını sağlar. Çok sınıflı sınıflandırma problemlerinde, softmax fonksiyonu genellikle modelin son katmanı olarak kullanılır. Bu sayede model, sınıflandırma işlemi için olasılık değerleri sunar ve en yüksek olasılığa sahip sınıfı tahmin eder.Böylelikle hangi hastalığı tahmin ettiğini buluyoruz. 
![image](https://github.com/user-attachments/assets/5f349794-5175-4960-8374-1b945f42e6e0)
Bu, modelin öğrenme sürecinde kullanılan optimizasyon algoritmasıdır.
Düşük bir öğrenme oranı, modelin eğitim sürecinin yavaş olmasına, ancak daha istikrarlı bir şekilde öğrenmesine yol açar. Yüksek bir öğrenme oranı ise daha hızlı öğrenmeye, ancak aşırı büyük adımlar atarak modelin optimum noktayı atlamasına neden olabilir. 
Kayıp fonksiyonu (loss function), modelin tahminlerinin ne kadar doğru olduğunu ölçer. Kategorik sınıflandırma problemlerinde categorical_crossentropy kayıp fonksiyonu yaygın olarak kullanılır. Bu fonksiyon, modelin tahmin ettiği sınıf olasılıklarını gerçek sınıflarla karşılaştırarak bir hata değeri üretir. 
metrics, modelin eğitim sürecinde hangi performans metriklerinin takip edileceğini belirler. Burada accuracy (doğruluk) metriği kullanılmıştır. Bu, modelin doğru sınıflandırma oranını hesaplar. 
Adam, hem momentum (momentum tabanlı öğrenme) hem de adapte edilebilir öğrenme oranları kullanan bir optimizasyon algoritmasıdır. Bu, daha hızlı ve verimli bir öğrenme süreci sağlar.
![image](https://github.com/user-attachments/assets/8c9dec82-c4bc-4647-ba90-4b7323ed92ae)
model.summary() fonksiyonu, modelin yapısının ve parametrelerinin genel bir özetini verir. Bu özet, modelin eğitimi için kullanılacak toplam parametre sayısını, her katmanın giriş ve çıkış şekillerini, ve her katmanın parametre sayısını görmek için kullanılır. Bu, modelin karmaşıklığını anlamak ve olası iyileştirmeleri görmek için önemli bir adımdır. 
![image](https://github.com/user-attachments/assets/1353d052-ab66-45de-a125-63e8184b7036)
Model yinelenerek daha düşük kayıp ve daha yüksek doğruluk oranlarının bulunduğu kısımdır. Eğitimimizin doğruluk ve kayıp değerleri gösterilmiştir. Doğrulanma doğruluk ve kayıp değerleri modelimizde genelleme yapmamızı sağlar.

![image](https://github.com/user-attachments/assets/3d3d7c30-fb96-4699-baf1-3b4e22d65628)

Modeli kaydederek her seferinde tekrar eğiterek başlamamızı önler. Modeli hızlı bir şekilde kullanabiliriz.

![image](https://github.com/user-attachments/assets/83b9a3b2-377b-4675-85d8-73717f97a736)
İlk koddaki  metrikler, modelin eğitim sürecinin nasıl ilerlediğini ve genelleme yeteneğini değerlendirmek için kullanılır. İkinci kodda, training_history.history verisini JSON formatında bir dosyaya yazmak için kullanılır. Modelin eğitim sürecindeki metriklerin kaydedilmesi, ileride inceleme veya paylaşma amaçlı faydalıdır. JSON formatı, verilerin başka sistemlere kolayca aktarılmasını sağlar. Bu şekilde elde edilen veriyi başka bir uygulama veya programla paylaşabilirsiniz. 

![image](https://github.com/user-attachments/assets/dde887e4-f878-46c4-b98b-e44a88d34c93)
Eğitim ve doğrulama doğruluğunu görselleştirmek için yazılmıştır.

![image](https://github.com/user-attachments/assets/89d10164-e71e-41a7-8f42-2462b8e91bd4)


![image](https://github.com/user-attachments/assets/1a149771-1b3a-4e8c-8042-5ba3fe270153)

Doğrulama veri setindeki sınıfları görmemizi sağlar.

![image](https://github.com/user-attachments/assets/9a7b349d-12d0-4d19-a586-be5918cd2f29)

![image](https://github.com/user-attachments/assets/23e3f514-25ba-4b82-97b4-1cda7b397504)

Test veri kümesi, modelin performansını değerlendirmek ve genelleme yeteneğini ölçmek için kullanılır. Bu veri kümesi, eğitim ve doğrulama veri kümelerinden farklıdır ve modelin hiç görmediği verilerden oluşur. 

![image](https://github.com/user-attachments/assets/7b3b0fcd-cd2b-4b1b-bb22-98252b3b6a0f)
    • y_pred: Modelin tahmin ettiği tüm sınıf olasılıklarını içerir.
    • predicted_categories: Modelin tahmin ettiği sınıf etiketlerinin sırasını içerir .
    • Bu bilgiler, test veri kümesindeki gerçek etiketlerle karşılaştırılarak modelin doğruluğu ve performansı değerlendirilebilir.

    ,1)model.predict(test_set):
    • Test veri kümesi için tahmin edilen olasılıkları hesaplar.
      
2)tf.argmax(y_pred, axis=1):
    • Her görüntü için en yüksek olasılığa sahip sınıfı seçer.

![image](https://github.com/user-attachments/assets/54a8c94b-556c-4e59-831d-a6a6837eeee0)

Birinci kod, test veri kümesindeki gerçek sınıf etiketlerini çıkarır ve modelin tahmin ettiği kategorilerle (predicted_categories) karşılaştırmaya hazır hale getirir. Burdaki amacımız en yüksek olasılıklı sınıfı bulmaktır.

İkinci kod,test veri kümesindeki her bir görüntünün gerçek sınıf etiketlerini içerir. 
Tahmin edilen etiketlerle (predicted_categories) karşılaştırılarak doğruluk ve diğer performans metriklerinin hesaplanmasında kritik bir role sahiptir. 

Üçüncü kod ise test veri kümesindeki her görüntü için modelin tahmin ettiği sınıfı içerir. 
Test veri kümesindeki gerçek etiketler (Y_true) ile karşılaştırılarak, modelin doğruluk, hassasiyet, duyarlılık gibi metrikleri hesaplanır.


![image](https://github.com/user-attachments/assets/958b80e5-3fea-40ca-9a14-78fe41f8603d
Modelin tahmin performansını analiz etmek için karışıklık matrisi (confusion matrix) oluşturulur. Karışıklık matrisi, tahmin edilen sınıflar ile gerçek sınıflar arasındaki ilişkiyi bir tablo şeklinde gösterir. Modelin hangi sınıflarda başarılı ya da başarısız olduğunu anlamak için kullanılan önemli bir araçtır. Karışıklık matrisi, sınıflar arasındaki karışıklığı sayısal olarak gösterir ve sınıflandırma performansını değerlendirmenize yardımcı olur. 

![image](https://github.com/user-attachments/assets/06de72df-9d8d-4fcf-9e9d-36fe9d55fe91)
Kod, classification_report fonksiyonunu kullanarak modelin her bir sınıf üzerindeki performansını ayrıntılı olarak değerlendiren bir rapor oluşturur. 
Precision: Modelin doğru tahmin yapma oranı. 
Recall: Modelin sınıfları doğru şekilde tanıyıp tanımadığı.
F1-Score: Precision ve Recall arasında denge kuran bir ölçüt.
Support: Test veri kümesindeki her sınıfın örnek sayısı.
