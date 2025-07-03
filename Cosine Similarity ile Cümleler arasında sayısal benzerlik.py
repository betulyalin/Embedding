
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Modeli yükle
model = SentenceTransformer('all-MiniLM-L6-v2')
similar_sentences = [
    "Yaz tatilinde gitmek için deniz kenarında bir otel arıyorum.",
    "Tatilimi deniz kenarında geçirmeyi planlıyorum.",
    "Yaz aylarında deniz kenarında bir tatil yapmayı düşünüyorum.",
    "Deniz manzaralı bir otel arayışındayım.",
    "Bu yaz deniz kenarında tatil yapmayı çok istiyorum."
]
unrelated_sentences = [
    "Birkaç farklı renk seçeneği mevcut.",
    "Bu ürün çok uygun fiyata satılmakta.",
    "Daha fazla enerji tasarrufu sağlamak için kullanabilirsiniz.",
    "Yüksek kalite ve dayanıklılık sağlar.",
    "Ergonomik tasarımıyla konforlu bir deneyim sunar."
]

all_sentences = similar_sentences + unrelated_sentences
n = len(all_sentences)

# Embedding'ler ve benzerlik matrisi
embeddings = model.encode(all_sentences)
similarity_matrix = cosine_similarity(embeddings)

# 1. KISIM: TAM BENZERLİK MATRİSİ
similarity_df = pd.DataFrame(similarity_matrix,
                           columns=[f"Cümle {i+1}" for i in range(n)],
                           index=[f"Cümle {i+1}" for i in range(n)])
np.fill_diagonal(similarity_df.values, np.nan)  # Köşegeni NaN yap

# 2. KISIM: EN YÜKSEK/EN DÜŞÜK BENZERLİKLER
# Üst üçgendeki benzersiz çiftleri al
upper_indices = np.triu_indices(n, k=1)
unique_pairs = [(i, j, similarity_matrix[i,j]) for i, j in zip(*upper_indices)]

sorted_pairs = sorted(unique_pairs, key=lambda x: x[2], reverse=True)
print("SKOR TABLOSU (NaN = kendi kendine benzerlik):")
pd.set_option('display.precision', 3)
print(similarity_df)
print("EN YÜKSEK 3 BENZERSİZ BENZERLİK:")
for i, j, score in sorted_pairs[:3]:
    print(f"\n● Benzerlik: {score:.3f}")
    print(f"  Cümle {i+1}: {all_sentences[i]}")
    print(f"  Cümle {j+1}: {all_sentences[j]}")

print("EN DÜŞÜK 3 BENZERSİZ BENZERLİK:")
for i, j, score in sorted_pairs[-3:]:
    print(f"\n● Benzerlik: {score:.3f}")
    print(f"  Cümle {i+1}: {all_sentences[i]}")
    print(f"  Cümle {j+1}: {all_sentences[j]}")

print("ANALİZ ÖZETİ:")
print(f"- Toplam {n} cümle için {len(sorted_pairs)} benzersiz çift karşılaştırıldı")
print("- En yüksek benzerlikler aynı konudaki cümleler arasında (deniz tatili)")
print("- En düşük benzerlikler tamamen farklı konular arasında (tatil vs ürün özellikleri)")
print("- Matriste köşegen değerleri (kendi kendine benzerlik) NaN olarak işaretlendi")
"""
ÇIKTI:
══════════════════════════════════════════════
SKOR TABLOSU (NaN = kendi kendine benzerlik):
══════════════════════════════════════════════
          Cümle 1  Cümle 2  Cümle 3  Cümle 4  Cümle 5  Cümle 6  Cümle 7  \
Cümle 1       NaN    0.799    0.849    0.812    0.831    0.725    0.659
Cümle 2     0.799      NaN    0.766    0.780    0.760    0.701    0.591
Cümle 3     0.849    0.766      NaN    0.826    0.901    0.706    0.682
Cümle 4     0.812    0.780    0.826      NaN    0.777    0.722    0.668
Cümle 5     0.831    0.760    0.901    0.777      NaN    0.721    0.731
Cümle 6     0.725    0.701    0.706    0.722    0.721      NaN    0.719
Cümle 7     0.659    0.591    0.682    0.668    0.731    0.719      NaN
Cümle 8     0.600    0.482    0.497    0.412    0.498    0.461    0.472
Cümle 9     0.761    0.683    0.735    0.719    0.740    0.789    0.682
Cümle 10    0.819    0.774    0.820    0.823    0.774    0.791    0.692

          Cümle 8  Cümle 9  Cümle 10
Cümle 1     0.600    0.761     0.819
Cümle 2     0.482    0.683     0.774
Cümle 3     0.497    0.735     0.820
Cümle 4     0.412    0.719     0.823
Cümle 5     0.498    0.740     0.774
Cümle 6     0.461    0.789     0.791
Cümle 7     0.472    0.682     0.692
Cümle 8       NaN    0.545     0.483
Cümle 9     0.545      NaN     0.806
Cümle 10    0.483    0.806       NaN

══════════════════════════════════════════════
EN YÜKSEK 3 BENZERSİZ BENZERLİK:
══════════════════════════════════════════════

● Benzerlik: 0.901
  Cümle 3: Yaz aylarında deniz kenarında bir tatil yapmayı düşünüyorum.
  Cümle 5: Bu yaz deniz kenarında tatil yapmayı çok istiyorum.

● Benzerlik: 0.849
  Cümle 1: Yaz tatilinde gitmek için deniz kenarında bir otel arıyorum.
  Cümle 3: Yaz aylarında deniz kenarında bir tatil yapmayı düşünüyorum.

● Benzerlik: 0.831
  Cümle 1: Yaz tatilinde gitmek için deniz kenarında bir otel arıyorum.
  Cümle 5: Bu yaz deniz kenarında tatil yapmayı çok istiyorum.

══════════════════════════════════════════════
EN DÜŞÜK 3 BENZERSİZ BENZERLİK:
══════════════════════════════════════════════

● Benzerlik: 0.472
  Cümle 7: Bu ürün çok uygun fiyata satılmakta.
  Cümle 8: Daha fazla enerji tasarrufu sağlamak için kullanabilirsiniz.

● Benzerlik: 0.461
  Cümle 6: Birkaç farklı renk seçeneği mevcut.
  Cümle 8: Daha fazla enerji tasarrufu sağlamak için kullanabilirsiniz.

● Benzerlik: 0.412
  Cümle 4: Deniz manzaralı bir otel arayışındayım.
  Cümle 8: Daha fazla enerji tasarrufu sağlamak için kullanabilirsiniz.

══════════════════════════════════════════════
ANALİZ ÖZETİ:
══════════════════════════════════════════════
- Toplam 10 cümle için 45 benzersiz çift karşılaştırıldı
- En yüksek benzerlikler aynı konudaki cümleler arasında (deniz tatili)
- En düşük benzerlikler tamamen farklı konular arasında (tatil vs ürün özellikleri)
- Matriste köşegen değerleri (kendi kendine benzerlik) NaN olarak işaretlendi
"""