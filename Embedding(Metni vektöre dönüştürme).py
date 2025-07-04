# -*- coding: utf-8 -*-
"""Untitled20.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AWMUx9R6hD5rNqeHxj-4jrJypJkEl3A5
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# SentenceTransformer modelini yükle
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

#Cümleleri embedding vektörlerine dönüştür
sentence_embeddings = model.encode(all_sentences)

first_5_values = sentence_embeddings[:5, :5]  # İlk 5 cümle için ilk 5 değeri

print("İlk 5 cümlenin ilk 5 vektör değeri:")
print(first_5_values)

# 5. Benzerlik hesaplama (Cosine similarity)
similarities = cosine_similarity(sentence_embeddings[:5], sentence_embeddings[5:])

print("\nBenzerlik Skorları:")
print(similarities)

"""
ÇIKTI:
İlk 5 cümlenin ilk 5 vektör değeri:
[[-0.06274065  0.07453643 -0.00829491 -0.00412108 -0.15439194]
 [-0.0016504   0.10951152 -0.00323487  0.00758793 -0.10836173]
 [-0.02580179  0.0193377   0.02140729  0.01589612 -0.14312111]
 [ 0.0191715   0.03392657 -0.02692877  0.02246912 -0.15848544]
 [ 0.00455694  0.04482879 -0.01908583  0.01387956 -0.10605039]]

Benzerlik Skorları:
[[0.72489727 0.658635   0.6004771  0.7613715  0.8191577 ]
 [0.70143163 0.5912824  0.48200026 0.68316734 0.77385455]
 [0.70619905 0.6823969  0.49683064 0.7354444  0.81972086]
 [0.72206295 0.6680676  0.41204283 0.7185006  0.82318735]
 [0.72120726 0.73149186 0.49830592 0.74023867 0.7737565 ]]
 """