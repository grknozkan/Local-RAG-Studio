# Building Your First Local RAG Application with Foundry Local

## 1. RAG (Retrieval-Augmented Generation) Nedir?
Geri Getirme Destekli Üretim (RAG), yapay zeka modellerinin kendi genel eğitim verileri yerine, doğrudan sizin verdiğiniz özel belgelere (şirket el kitapları, PDF'ler, kod depoları, yerel dosyalar) dayanarak yanıt üretmesini sağlayan yenilikçi bir yöntemdir.

## 2. Local (Yerel) ve Çevrimdışı Çalışmanın Avantajları
- **%100 Veri Gizliliği**: Özel belgeleriniz ve sorularınız hiçbir dış sunucuya (OpenAI, Anthropic vb.) gönderilmez.
- **Sıfır API Maliyeti**: Bulut yapay zeka servislerine aylık abonelik veya jeton (token) ücreti ödemezsiniz.
- **Kesintisiz Çevrimdışı Erişim**: İnternet bağlantınız olmasa bile uçakta, sahada veya güvenlik kısıtlaması olan ağlarda sorunsuz çalışır.

## 3. Microsoft Foundry Local ve Ollama Entegrasyonu
Microsoft Foundry Local ve Ollama, açık kaynaklı dil modellerini (`llama3`, `mistral`, `qwen2.5`, `phi3`) tek bir terminal komutuyla bilgisayarınızda bir yerel API sunucusu olarak çalıştırmanızı sağlar.

### Ollama Kurulum ve Başlatma Komutları:
```bash
# Llama 3 modelini indirme ve çalıştırma
ollama run llama3

# Mistral modelini indirme
ollama run mistral

# Qwen 2.5 Türkçe destekli model
ollama run qwen2.5
```

## 4. Vektör Veritabanı ve Benzerlik Araması (Cosine Similarity)
1. **Parçalama (Chunking)**: Büyük belgeler küçük metin bloklarına ayrılır (örneğin 500 karakterlik parçalar).
2. **Vektörleştirme (Embedding)**: Metin parçaları anlamsal matrislere dönüştürülür.
3. **Top-K Arama**: Kullanıcı soru sorduğunda, Kosinüs Benzerliği (Cosine Similarity) formülü kullanılarak en yüksek uyum gösteren ilk K parça (Top-K) bulunur ve yapay zekaya bağlam olarak aktarılır.
