# 🚀 Foundry Local RAG Studio

Foundry Local RAG Studio, **%100 kendi bilgisayarınızda (çevrimdışı / offline)** çalışan, verilerinizin dışarıya gitmediği ve bulut servislerine bağımlı olmayan gelişmiş bir **Retrieval-Augmented Generation (RAG)** uygulamasıdır.

---

## 🌟 Öne Çıkan Özellikler

- **🔒 %100 Yerel Veri Gizliliği**: Yüklediğiniz belgeler ve sorduğunuz sorular sadece bilgisayarınızda işlenir.
- **⚡ Dahili Çevrimdışı Vektör Motoru**: Herhangi bir dış bağımlılık kurmadan doğrudan tarayıcıda metin parçalama (chunking), TF-IDF & Cosine Similarity vektör araması yapar.
- **🤖 Microsoft Foundry Local & Ollama Desteği**: Bilgisayarınızda çalışan Ollama (`llama3`, `mistral`, `qwen2.5`, `phi3`) modelleriyle doğrudan entegre olur.
- **📊 Görsel RAG Akış Takibi (Pipeline Visualizer)**: Sorgunuzun geçtiği 5 aşamayı canlı olarak görselleştirir:
  1. *Belge Parçalama (Chunking)*
  2. *Vektör Embeddings*
  3. *Top-K Benzerlik Arama*
  4. *Prompt Artırımı (Augmentation)*
  5. *Yerel YZ Yanıt Üretimi*
- **📌 Kaynak Gösterimi (Citations)**: Yapay zekanın yanıtlarını hangi belgenin hangi parçasından aldığını % uyum skoru ile gösterir.
- **🎛️ Canlı Parametre Ayarları**: Parça Boyutu (Chunk Size), Çakışma (Overlap), Top-K ve Benzerlik Eşiğini canlı olarak değiştirebilirsiniz.

---

## 🚀 Hızlı Başlangıç

### Yöntem 1: Doğrudan Tarayıcıda Açma (Sıfır Kurulum)
1. `index.html` dosyasını herhangi bir web tarayıcısında (Chrome, Edge, Firefox) çift tıklayarak açın.
2. Sol paneldeki **"Örnek Belgeleri Yükle"** butonuna basarak demoyu hemen başlatın.
3. Sorunuzu yazın ve yerel RAG akışını canlı izleyin!

### Yöntem 2: Local HTTP Sunucusu ile Çalıştırma
Proje dizininde basit bir Python veya Node sunucusu başlatabilirsiniz:

```bash
# Python ile:
python -m http.server 3000

# Veya Node.js ile:
npx serve .
```
Tarayıcınızda `http://localhost:3000` adresine gidin.

---

## 🦙 Ollama / Foundry Local Bağlantısı (Opsiyonel)

Bilgisayarınızda yerel bir LLM çalıştırmak isterseniz:

1. [Ollama](https://ollama.com) uygulamasını indirin veya başlatın.
2. Terminalinizden istediğiniz modeli indirin:
   ```bash
   ollama run llama3
   # veya Türkçe destekli:
   ollama run qwen2.5
   ```
3. Uygulamadaki sağ üst **Model Seçici** menüsünden `llama3` veya `qwen2.5` modelini seçin.
4. Uygulama otomatik olarak `http://localhost:11434` adresine bağlanacaktır!

---

## 📁 Proje Yapısı

```
local-rag-foundry/
├── index.html                  # Ana Kullanıcı Arayüzü & RAG Dashboard
├── style.css                   # Modern Neon/Dark Glassmorphism Stil Sistemi
├── app.js                      # Vektör Arama Motoru, Pipeline Görselleştirici & Ollama Entegrasyonu
├── package.json                # Proje Bağımlılıkları ve Yapılandırma
├── sample_documents/           # Test için Örnek Belgeler
│   └── Foundry_Local_RAG_Rehberi.md
└── README.md                   # Kullanım ve Mimari Rehberi
```

---

## 💡 RAG Mimarlık Notları

```
[Belgeler] ➔ [Parçalama / Chunking] ➔ [TF-IDF Vektör İndeksi]
                                                │
[Kullanıcı Sorusu] ➔ [Cosine Similarity] ───────┘
                           │
                 [Top-K Alakalı Parçalar]
                           │
                 [Artırılmış Prompt] ➔ [Ollama / Yerel YZ] ➔ [Yanıt & Kaynaklar]
```
