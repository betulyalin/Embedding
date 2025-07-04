!pip install fsspec==2025.3.0

!pip install -q -U transformers datasets peft accelerate bitsandbytes sentence-transformers faiss-cpu

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# BERT tokenizer ve modelini yükle
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Örnek ürün açıklamaları
product_descriptions = [
    "Kablosuz elektrikli diş fırçası: yüksek frekanslı titreşimleri ile dişlerinizi derinlemesine temizler. Su geçirmez tasarımı, uzun pil ömrü ve farklı temizleme modlarıyla ağız sağlığınızı en üst düzeye çıkarır.",
    "Yoga Matı: Kaymaz yüzeyi, 6 mm kalınlığı ve çevre dostu malzemesi ile bu yoga matı, egzersiz yaparken rahatlık ve güvenlik sağlar. Hem evde hem de dış mekanlarda kullanıma uygun, hafif ve taşınabilir.",
    "Beyaz Eşya Seti: Mikrodalga, bulaşık makinesi, çamaşır makinesi ve buzdolabından oluşan bu beyaz eşya seti, modern mutfaklar için mükemmel bir çözüm sunar. Şık tasarımı ve enerji verimliliği ile her ihtiyacınızı karşılar.",
    "Akıllı Termostat: Evinizdeki sıcaklık seviyesini istediğiniz gibi ayarlayabileceğiniz bu akıllı termostat, enerji tasarrufu sağlamak ve yaşam alanlarınızı daha konforlu hale getirmek için gelişmiş sensörler ve mobil uygulama desteği sunar.",
    "Elektrikli Isıtıcı: Kompakt ve hafif tasarımıyla bu elektrikli ısıtıcı, küçük alanlarda hızlı ve etkili ısıtma sağlar. 3 farklı ısıtma modu ve güvenlik kilidi özellikleri ile kullanımda güvenliği ön planda tutar.",
    "Bisiklet: Ergonomik tasarımı, 21 vitesli şanzımanı ve dayanıklı lastikleriyle bu bisiklet, şehir içi ulaşımda ve doğa sporlarında mükemmel bir tercihtir. Hem kısa mesafelerde hem de uzun yolculuklarda rahat bir sürüş sunar.",
    "Sırt Çantası: Bu şık ve fonksiyonel sırt çantası, laptop bölmesi, su şişesi cebi ve çok sayıda cephesi ile günlük kullanım için idealdir. Sağlam kumaşı ve ayarlanabilir askıları sayesinde uzun süre konforlu bir kullanım sunar.",
    "Televizyon: 4K çözünürlük, HDR desteği ve 55 inç ekranı ile sinema keyfini evinize getirir. HDMI ve USB bağlantı noktaları ile çeşitli cihazlarla uyumludur.",
    "Tartışmasız İçecek Karıştırıcı: Meyve ve sebzeleri karıştırarak sağlıklı içecekler hazırlamanızı sağlayan bu blender, 5 farklı hız ayarına sahiptir. Sağlıklı yaşam tarzını benimseyenler için ideal bir mutfak aracı.",
    "Gözlük: Şık tasarımı ve UV400 filtreli lensleri ile bu gözlük, güneş ışınlarının zararlı etkilerinden korunmanıza yardımcı olur. Rahat kullanım için ergonomik yapısı ve hafif çerçevesi ile günlük kullanım için idealdir."
]

# Sorgu cümlesi
query = "Sağlıklı yaşam için uygun bir ürün arıyorum."

# Tokenize işlemi
inputs = tokenizer([query] * len(product_descriptions), product_descriptions, return_tensors='pt', padding=True, truncation=True)

# Model ile benzerlik hesaplama
outputs = model(**inputs)
logits = outputs.logits

# Softmax ile benzerlik hesaplama
similarity = torch.softmax(logits, dim=-1)

# Benzerlik skorlarını al ve en yüksek 3 sonucu seç
similarity_scores = similarity[:, 1].detach().numpy()  # Sağlıklı yaşam kategorisindeki skorları al
top_n_indices = similarity_scores.argsort()[-3:][::-1]  # En yüksek 3 skoru al

# 3 En yakın sonucu al
top_n_results = [product_descriptions[i] for i in top_n_indices]

# Sonuçları yazdır
print("Top-N yakın sonuçlar:")
for idx, result in enumerate(top_n_results, 1):
    print(f"{idx}. {result}")
------------------------------------------------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Modeli yükle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ürün açıklamaları (örnek)
product_descriptions = [
    "Kablosuz elektrikli diş fırçası: yüksek frekanslı titreşimleri ile dişlerinizi derinlemesine temizler. Su geçirmez tasarımı, uzun pil ömrü ve farklı temizleme modlarıyla ağız sağlığınızı en üst düzeye çıkarır.",
    "Yoga Matı: Kaymaz yüzeyi, 6 mm kalınlığı ve çevre dostu malzemesi ile bu yoga matı, egzersiz yaparken rahatlık ve güvenlik sağlar. Hem evde hem de dış mekanlarda kullanıma uygun, hafif ve taşınabilir.",
    "Beyaz Eşya Seti: Mikrodalga, bulaşık makinesi, çamaşır makinesi ve buzdolabından oluşan bu beyaz eşya seti, modern mutfaklar için mükemmel bir çözüm sunar. Şık tasarımı ve enerji verimliliği ile her ihtiyacınızı karşılar.",
    "Akıllı Termostat: Evinizdeki sıcaklık seviyesini istediğiniz gibi ayarlayabileceğiniz bu akıllı termostat, enerji tasarrufu sağlamak ve yaşam alanlarınızı daha konforlu hale getirmek için gelişmiş sensörler ve mobil uygulama desteği sunar.",
    "Elektrikli Isıtıcı: Kompakt ve hafif tasarımıyla bu elektrikli ısıtıcı, küçük alanlarda hızlı ve etkili ısıtma sağlar. 3 farklı ısıtma modu ve güvenlik kilidi özellikleri ile kullanımda güvenliği ön planda tutar.",
    "Bisiklet: Ergonomik tasarımı, 21 vitesli şanzımanı ve dayanıklı lastikleriyle bu bisiklet, şehir içi ulaşımda ve doğa sporlarında mükemmel bir tercihtir. Hem kısa mesafelerde hem de uzun yolculuklarda rahat bir sürüş sunar.",
    "Sırt Çantası: Bu şık ve fonksiyonel sırt çantası, laptop bölmesi, su şişesi cebi ve çok sayıda cephesi ile günlük kullanım için idealdir. Sağlam kumaşı ve ayarlanabilir askıları sayesinde uzun süre konforlu bir kullanım sunar.",
    "Televizyon: 4K çözünürlük, HDR desteği ve 55 inç ekranı ile sinema keyfini evinize getirir. HDMI ve USB bağlantı noktaları ile çeşitli cihazlarla uyumludur.",
    "Tartışmasız İçecek Karıştırıcı: Meyve ve sebzeleri karıştırarak sağlıklı içecekler hazırlamanızı sağlayan bu blender, 5 farklı hız ayarına sahiptir. Sağlıklı yaşam tarzını benimseyenler için ideal bir mutfak aracı.",
    "Gözlük: Şık tasarımı ve UV400 filtreli lensleri ile bu gözlük, güneş ışınlarının zararlı etkilerinden korunmanıza yardımcı olur. Rahat kullanım için ergonomik yapısı ve hafif çerçevesi ile günlük kullanım için idealdir."
]

# Ürün açıklamalarını embedding'lere dönüştür
product_embeddings = model.encode(product_descriptions)

# FAISS index oluştur
dimension = product_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# FAISS index'e vektörleri ekle
index.add(np.array(product_embeddings))

# Sorgu al
query = "Sağlıklı yaşam için uygun bir ürün arıyorum"

# Sorguyu embedding'e dönüştür
query_embedding = model.encode([query])

# Normalize et (Cosine Similarity hesaplamak için)
product_embeddings = product_embeddings / np.linalg.norm(product_embeddings, axis=1, keepdims=True)
query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

# Top-N yakın sonuçları bul
N = 3
D, I = index.search(np.array(query_embedding), k=N)

# Sonuçları yazdır
top_n_results = [product_descriptions[i] for i in I[0]]
print("Top-N yakın sonuçlar:")
for idx, result in enumerate(top_n_results, 1):
    print(f"{idx}. {result}")
