# Depo Yeri Seçimi Problemi (WLP) için Hibrit Yaklaşım

**K-Means Kümeleme ve Açgözlü (Greedy) Algoritma Entegrasyonu**

> Manisa Celal Bayar Üniversitesi — Hasan Ferdi Turgutlu Teknoloji Fakültesi
> Yazılım Mühendisliği Bölümü
> **Algoritma Analizi ve Tasarımı** Dersi Dönem Projesi
> 2025–2026 Bahar Dönemi

---

## 👥 Yazarlar

- **Ömer Faruk DAMAR**
- **Mehmet Ali AVCI**

---

## 📖 Proje Hakkında

Bu proje, NP-zor sınıfındaki **Depo Yeri Seçimi Problemi (Warehouse Location Problem - WLP)** üzerinde K-Means kümeleme ve Açgözlü (Greedy) atama yaklaşımlarını birleştiren özgün bir **hibrit algoritma** önermektedir. Önerilen yöntem, kapasite ihlallerini gidermek için takas tabanlı yerel arama adımı da içermektedir.

Algoritmanın performansı üç farklı ölçek düzeyinde (50, 100 ve 200 müşteri) ve farklı $K$ değerleri için ($K \in \{2,\ldots,8\}$):

- ✅ **Greedy-Only** (Saf Açgözlü)
- ✅ **Random Search** (Rastgele Arama)
- ✅ **Integer Programming (IP)** (Tam Sayılı Programlama)

ile karşılaştırmalı olarak değerlendirilmiştir.

---

## 🗂️ Proje Yapısı

```
.
├── wlp_final.py              # Tüm algoritmaların Python implementasyonu
├── bildiri_final.tex         # LaTeX akademik bildiri kaynak dosyası
├── bildiri_final.pdf         # Derlenmiş PDF çıktı
├── fig_cost_curves.png       # Maliyet karşılaştırma grafikleri
├── fig_time_bars.png         # Çalışma süresi karşılaştırması
├── fig_gap_heatmap.png       # Optimality gap ısı haritası
├── fig_solution_map.png      # Depo seçim haritaları (K=5)
├── fig_scalability.png       # Ölçeklenebilirlik analizi
└── README.md
```

---

## 🚀 Çalıştırma

### Gereksinimler

```bash
pip install numpy matplotlib scikit-learn pulp
```

### Python kodunu çalıştırma

```bash
python wlp_final.py
```

Çalıştığında:
- 3 farklı veri setinde (Küçük, Orta, Büyük) tüm algoritmaları test eder
- $K = 2, 3, \ldots, 8$ için karşılaştırmalı sonuçları konsola yazdırır
- 5 adet grafik (PNG) üretir

### LaTeX bildirisini derleme

[Overleaf](https://www.overleaf.com) üzerinde projeyi açıp `bildiri_final.tex` dosyasını derle veya yerel olarak:

```bash
pdflatex bildiri_final.tex
```

---

## 🧠 Önerilen Hibrit Algoritma

Üç aşamadan oluşur:

1. **K-Means Kümeleme:** Müşteri konumları $K$ adet kümeye ayrılır.
2. **Greedy Depo Seçimi:** Her küme merkezine en yakın henüz seçilmemiş aday depo seçilir.
3. **Yerel İyileştirme (Swap):** Kapasite ihlali varsa tek depo takasıyla iyileştirme yapılır.

### Sözde Kod

```
S ← ∅
{C₁, ..., C_K} ← KMeans(C, K)
for k = 1 to K:
    p_k ← Centroid(C_k)
    j* ← argmin {||p_k - w_j||₂ : j ∈ W \ S}
    S ← S ∪ {j*}
while improvement possible:
    (out, in) ← BestSwap(S, W \ S)
    if Cost(S \ {out} ∪ {in}) < Cost(S):
        S ← S \ {out} ∪ {in}
return S
```

---

## 📊 Önemli Bulgular

- Hibrit yöntem, **saf Açgözlü algoritmayı** tüm veri setlerinde belirgin biçimde geride bırakmıştır.
- Orta ve büyük ölçekli problemlerde hibrit yöntem, **Rastgele Arama'yı** $K \geq 4$ için açıkça geride bırakmıştır.
- **IP** en düşük maliyeti garantilemekte; ancak yüksek çalışma süresi büyük örneklerde kullanımı kısıtlamaktadır.
- Hibrit yöntemin ortalama çalışma süresi **8–90 ms** aralığında kalmakta; gerçek zamanlı karar destek sistemleri için uygun bir hesaplama profili sunmaktadır.

---

## 📚 Bildiri (Paper)

Çalışmanın detaylı sonuçları, IEEE Conference Template formatında hazırlanmış akademik bildiride yer almaktadır:

- 📄 [bildiri_final.pdf](./bildiri_final.pdf)

### Bildiri Bölümleri

- **Abstract** (Özet)
- **Introduction** (Giriş)
- **Literature Review** (Literatür Taraması)
- **Method** (Yöntem)
- **Experimental Results** (Deneysel Sonuçlar)
- **Conclusion and Future Work** (Sonuç ve Gelecek Çalışmalar)

---

## 🛠️ Kullanılan Teknolojiler

- **Python 3.11**
- **NumPy** — Sayısal hesaplamalar
- **scikit-learn** — K-Means kümeleme
- **PuLP / CBC** — Tam Sayılı Programlama (MIP) çözümü
- **Matplotlib** — Grafik üretimi
- **LaTeX (IEEE Conference Template)** — Bildiri formatı

---

## 📜 Lisans

Bu proje akademik amaçlı geliştirilmiştir.

---

## 📬 İletişim

Proje hakkında sorularınız için issue açabilirsiniz.
