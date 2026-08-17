/**
 * Foundry Local RAG Studio - Core Engine & Visualizer
 * Fully local, offline capable Retrieval-Augmented Generation system.
 */

// ==========================================================================
// RAG State & Database Storage
// ==========================================================================
const state = {
  documents: [],          // [{ id, name, type, text, size, uploadedAt }]
  chunks: [],             // [{ id, docId, docName, index, text, vector, wordMap }]
  vocabulary: new Set(),  // Unique word vocabulary for TF-IDF vectors
  idfMap: new Map(),      // Inverse Document Frequency for terms
  
  // Pipeline Tuning Config
  config: {
    chunkSize: 500,
    chunkOverlap: 100,
    topK: 3,
    minScore: 20,
    temperature: 0.3,
    endpointUrl: 'http://localhost:11434',
    selectedModel: 'builtin-offline',
    customModelName: 'llama3',
    mode: 'builtin'       // 'ollama' or 'builtin'
  },

  // Connection State
  ollamaAvailable: false,
  availableModels: []
};

// Sample Pre-loaded Knowledge Base Documents for Instant RAG Demo
const sampleDocuments = [
  {
    id: 'sample-foundry-guide',
    name: 'Foundry_Local_RAG_Rehberi.md',
    type: 'text/markdown',
    size: '4.2 KB',
    uploadedAt: new Date().toLocaleTimeString(),
    text: `# Building Your First Local RAG Application with Foundry Local

## 1. Giriş ve RAG Konsepti
Retrieval-Augmented Generation (RAG / Geri Getirme Destekli Üretim), yapay zekanın sadece genel internet verisine değil, doğrudan şirket el kitapları, PDF'ler ve özel belgelere dayanarak yanıt vermesini sağlar.

## 2. Local (Yerel) ve Gizlilik
Local RAG mimarisinde yapay zeka modelleri ve vektör veritabanı doğrudan geliştiricinin kendi bilgisayarında (laptop veya masaüstü) çalışır. Verileriniz asla dışarıdaki bir bulut sunucusuna gitmez, internet bağlantısı gerektirmez ve %100 gizlilik sağlar.

## 3. Microsoft Foundry Local ve Ollama
Foundry Local ve Ollama, Llama 3, Mistral, Qwen 2.5 ve Phi-3 gibi açık kaynaklı LLM'leri ek kurulum zorluğu olmadan yerel bilgisayarınızda çalıştırmanızı sağlar.

## 4. Vektör Veritabanı ve Benzerlik Araması
Belgeler parça parça (chunk) metinlere bölünür. Her metin parçası anlamsal matrise (embedding) çevrilir. Kullanıcı bir soru sorduğunda Cosine Similarity (Kosinüs Benzerliği) ile en alakalı Top-K parça bulunur.

## 5. Avantajlar
- Bulut API ücreti yoktur ($0 maliyet).
- Güvenlik ve KVKK / GDPR uyumluluğu mükemmeldir.
- Çevrimdışı (offline) uçakta veya şantiyede dahi kesintisiz çalışır.`
  },
  {
    id: 'sample-company-handbook',
    name: 'Sirket_Ici_Calisma_Rehberi.txt',
    type: 'text/plain',
    size: '3.8 KB',
    uploadedAt: new Date().toLocaleTimeString(),
    text: `AKILLI TEKNOLOJİLER ŞİRKET İÇİ ÇALIŞMA VE GÜVENLİK YÖNERGESİ

1. ÇALIŞMA SAATLERİ VE UZAKTAN ÇALIŞMA
Şirketimizde esnek çalışma saatleri uygulanmaktadır. Çekirdek çalışma saatleri 10:00 - 16:00 arasındadır. Çalışanlar haftada 2 gün uzaktan (home-office) çalışma hakkına sahiptir.

2. İZİN POLİTİKASI VE YILLIK İZİN
1 yılını dolduran çalışanların 14 iş günü yıllık izin hakkı bulunur. İzin talepleri Portal üzerinden en az 5 gün öncesinden yönetici onayına sunulmalıdır.

3. SİBER GÜVENLİK VE YAPAY ZEKA KULLANIMI
Şirket verileri, müşteri bilgileri ve kaynak kodlar kesinlikle halka açık üçüncü taraf bulut yapay zeka servislerine (ChatGPT, Claude vb.) yapıştırılamaz. 
Yerel belgeler üzerinde işlem yapmak için şirket tarafından onaylanmış %100 yerel ve çevrimdışı Foundry Local RAG sistemleri kullanılmalıdır.

4. VPN VE ŞİRKET EKİPMANLARI
Şirket laptopları sadece şirket işleri için kullanılır. Dış ağlardan şirket içi sunuculara erişirken FortiClient VPN bağlantısı zorunludur.`
  }
];

// ==========================================================================
// Initialization & Event Listeners
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  initUIEvents();
  initPDFJS();
  checkOllamaHealth();
  
  // Load sample documents automatically for instant demonstration
  loadSampleDocuments();
});

// PDF.js worker setup
function initPDFJS() {
  if (window.pdfjsLib) {
    window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  }
}

function initUIEvents() {
  // Config Sliders Bindings
  bindSlider('chunkSize', 'chunkSizeVal', ' karakter');
  bindSlider('chunkOverlap', 'chunkOverlapVal', ' karakter');
  bindSlider('topK', 'topKVal', ' Parça');
  bindSlider('minScore', 'minScoreVal', '%');
  bindSlider('temperature', 'tempVal', '');

  // Reset Config
  document.getElementById('btnResetConfig').addEventListener('click', () => {
    document.getElementById('chunkSize').value = 500;
    document.getElementById('chunkOverlap').value = 100;
    document.getElementById('topK').value = 3;
    document.getElementById('minScore').value = 20;
    document.getElementById('temperature').value = 0.3;
    
    updateSliderDisplays();
    rebuildVectorIndex();
  });

  // Dropzone & File Upload
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files);
    }
  });

  // Load Samples Button
  document.getElementById('btnLoadSamples').addEventListener('click', loadSampleDocuments);

  // Chat Form Submission
  document.getElementById('chatForm').addEventListener('submit', (e) => {
    e.preventDefault();
    submitUserQuery();
  });

  // Enter to Submit in Textarea
  document.getElementById('userInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitUserQuery();
    }
  });

  // Quick Prompt Templates
  document.querySelectorAll('.template-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const promptText = btn.getAttribute('data-prompt');
      document.getElementById('userInput').value = promptText;
      submitUserQuery();
    });
  });

  // Model Selector Change
  document.getElementById('modelSelector').addEventListener('change', (e) => {
    state.config.selectedModel = e.target.value;
    if (e.target.value === 'builtin-offline') {
      state.config.mode = 'builtin';
    } else {
      state.config.mode = 'ollama';
      state.config.customModelName = e.target.value;
    }
    updateActiveModelFooter();
  });

  // Settings Modal Controls
  document.getElementById('btnSettings').addEventListener('click', openSettingsModal);
  document.getElementById('btnCloseModal').addEventListener('click', closeSettingsModal);
  document.getElementById('btnCancelModal').addEventListener('click', closeSettingsModal);
  document.getElementById('btnSaveModal').addEventListener('click', saveSettings);
  document.getElementById('btnTestConn').addEventListener('click', testConnectionInModal);

  // Inspector Tabs
  document.querySelectorAll('.inspector-tabs .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.inspector-tabs .tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      if (tabId === 'retrievedChunks') {
        document.getElementById('tabRetrievedChunks').classList.add('active');
      } else {
        document.getElementById('tabAugmentedPrompt').classList.add('active');
      }
    });
  });
}

function bindSlider(id, valId, suffix) {
  const slider = document.getElementById(id);
  slider.addEventListener('input', () => {
    document.getElementById(valId).textContent = slider.value + suffix;
    state.config[id] = parseFloat(slider.value);
    
    if (id === 'chunkSize' || id === 'chunkOverlap') {
      rebuildVectorIndex();
    }
  });
}

function updateSliderDisplays() {
  document.getElementById('chunkSizeVal').textContent = document.getElementById('chunkSize').value + ' karakter';
  document.getElementById('chunkOverlapVal').textContent = document.getElementById('chunkOverlap').value + ' karakter';
  document.getElementById('topKVal').textContent = document.getElementById('topK').value + ' Parça';
  document.getElementById('minScoreVal').textContent = document.getElementById('minScore').value + '%';
  document.getElementById('tempVal').textContent = document.getElementById('temperature').value;
  
  state.config.chunkSize = parseInt(document.getElementById('chunkSize').value);
  state.config.chunkOverlap = parseInt(document.getElementById('chunkOverlap').value);
  state.config.topK = parseInt(document.getElementById('topK').value);
  state.config.minScore = parseInt(document.getElementById('minScore').value);
  state.config.temperature = parseFloat(document.getElementById('temperature').value);
}

// ==========================================================================
// Document Ingestion & File Processing
// ==========================================================================
async function handleFileUpload(files) {
  for (let file of files) {
    const ext = file.name.split('.').pop().toLowerCase();
    let text = '';

    try {
      if (ext === 'pdf') {
        text = await extractTextFromPDF(file);
      } else {
        text = await readTextFile(file);
      }

      if (text.trim().length > 0) {
        addDocumentToState({
          id: 'doc-' + Date.now() + '-' + Math.random().toString(36).substr(2, 4),
          name: file.name,
          type: file.type || ext,
          size: formatBytes(file.size),
          uploadedAt: new Date().toLocaleTimeString(),
          text: text
        });
      }
    } catch (err) {
      console.error('Dosya okuma hatası:', err);
      alert(`"${file.name}" okunamadı: ` + err.message);
    }
  }
}

function readTextFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (e) => reject(e);
    reader.readAsText(file);
  });
}

async function extractTextFromPDF(file) {
  if (!window.pdfjsLib) {
    throw new Error('PDF.js kütüphanesi yüklenemedi.');
  }

  const arrayBuffer = await file.arrayBuffer();
  const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  let fullText = '';

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items.map(item => item.str).join(' ');
    fullText += `--- Sayfa ${i} ---\n` + pageText + '\n\n';
  }

  return fullText;
}

function loadSampleDocuments() {
  sampleDocuments.forEach(doc => {
    if (!state.documents.some(d => d.id === doc.id)) {
      addDocumentToState(doc);
    }
  });
}

function addDocumentToState(doc) {
  state.documents.push(doc);
  renderDocumentList();
  rebuildVectorIndex();
}

function removeDocument(docId) {
  state.documents = state.documents.filter(d => d.id !== docId);
  renderDocumentList();
  rebuildVectorIndex();
}

function renderDocumentList() {
  const docList = document.getElementById('docList');
  const badge = document.getElementById('docCountBadge');
  
  badge.textContent = `${state.documents.length} Belge`;

  if (state.documents.length === 0) {
    docList.innerHTML = `
      <div class="empty-state">
        <i class="fa-solid fa-file-circle-plus"></i>
        <p>Henüz yüklenmiş belge yok</p>
      </div>`;
    return;
  }

  docList.innerHTML = state.documents.map(doc => `
    <div class="doc-item">
      <div class="doc-item-info">
        <i class="fa-solid ${getFileIcon(doc.name)}"></i>
        <div>
          <div class="doc-name" title="${doc.name}">${doc.name}</div>
          <div class="doc-meta">${doc.size} • ${doc.uploadedAt}</div>
        </div>
      </div>
      <div class="doc-actions">
        <button title="Belgeyi Sil" onclick="removeDocument('${doc.id}')">
          <i class="fa-solid fa-trash-can"></i>
        </button>
      </div>
    </div>
  `).join('');
}

function getFileIcon(filename) {
  if (filename.endsWith('.pdf')) return 'fa-file-pdf';
  if (filename.endsWith('.md')) return 'fa-file-code';
  if (filename.endsWith('.json')) return 'fa-file-lines';
  return 'fa-file-lines';
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// ==========================================================================
// RAG Vector Engine & Chunking
// ==========================================================================
function rebuildVectorIndex() {
  triggerStep(1, 'active');
  
  // 1. Text Chunking
  const chunkSize = state.config.chunkSize;
  const overlap = state.config.chunkOverlap;
  state.chunks = [];
  state.vocabulary.clear();
  state.idfMap.clear();

  state.documents.forEach(doc => {
    const textChunks = chunkText(doc.text, chunkSize, overlap);
    textChunks.forEach((chunkTextStr, idx) => {
      state.chunks.push({
        id: `chunk-${doc.id}-${idx}`,
        docId: doc.id,
        docName: doc.name,
        index: idx + 1,
        text: chunkTextStr,
        tokens: tokenize(chunkTextStr),
        vector: null
      });
    });
  });

  // 2. Build Vocabulary & Calculate TF-IDF
  triggerStep(1, 'completed');
  triggerStep(2, 'active');

  const totalChunks = state.chunks.length;
  const docFreqMap = new Map();

  state.chunks.forEach(chunk => {
    const uniqueTokens = new Set(chunk.tokens);
    uniqueTokens.forEach(token => {
      state.vocabulary.add(token);
      docFreqMap.set(token, (docFreqMap.get(token) || 0) + 1);
    });
  });

  // Compute IDF for each term
  docFreqMap.forEach((count, token) => {
    state.idfMap.set(token, Math.log((totalChunks + 1) / (count + 1)) + 1);
  });

  // Compute Term Vector for each chunk
  state.chunks.forEach(chunk => {
    chunk.vector = createTFIDFVector(chunk.tokens);
  });

  triggerStep(2, 'completed');
  updateStatsDisplay();
}

function chunkText(text, size, overlap) {
  if (!text || text.trim().length === 0) return [];
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    let end = start + size;
    if (end < text.length) {
      // Find clean word / sentence / paragraph break to avoid cutting words
      const lastBreak = text.lastIndexOf('\n', end);
      if (lastBreak > start + size * 0.4) {
        end = lastBreak;
      } else {
        const lastSentence = text.lastIndexOf('. ', end);
        if (lastSentence > start + size * 0.4) {
          end = lastSentence + 1;
        } else {
          const lastSpace = text.lastIndexOf(' ', end);
          if (lastSpace > start) {
            end = lastSpace;
          }
        }
      }
    }

    const chunkStr = text.substring(start, end).trim();
    if (chunkStr.length > 0) {
      chunks.push(chunkStr);
    }

    start = end - overlap;
    if (start > 0 && start < text.length) {
      const prevSpace = text.lastIndexOf(' ', start);
      if (prevSpace > start - 40 && prevSpace > 0) {
        start = prevSpace + 1;
      }
    }

    if (start >= text.length - overlap) break;
  }

  return chunks;
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s\u00C0-\u024F\u1E00-\u1EFFğüşıöçĞÜŞİÖÇ]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 1);
}

function createTFIDFVector(tokens) {
  const termCounts = new Map();
  tokens.forEach(token => {
    termCounts.set(token, (termCounts.get(token) || 0) + 1);
  });

  const vector = new Map();
  const totalTokens = tokens.length || 1;

  termCounts.forEach((count, token) => {
    const tf = count / totalTokens;
    const idf = state.idfMap.get(token) || 1;
    vector.set(token, tf * idf);
  });

  return vector;
}

function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  vecA.forEach((valA, key) => {
    normA += valA * valA;
    if (vecB.has(key)) {
      dotProduct += valA * vecB.get(key);
    }
  });

  vecB.forEach(valB => {
    normB += valB * valB;
  });

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function retrieveTopK(queryText, k, minScorePercent) {
  triggerStep(3, 'active');
  
  const queryTokens = tokenize(queryText);
  const queryVector = createTFIDFVector(queryTokens);
  
  const scoredChunks = state.chunks.map(chunk => {
    const score = cosineSimilarity(queryVector, chunk.vector);
    return {
      chunk: chunk,
      score: score,
      scorePercent: Math.round(score * 100)
    };
  });

  scoredChunks.sort((a, b) => b.score - a.score);

  const filtered = scoredChunks.filter(item => item.scorePercent >= minScorePercent).slice(0, k);
  triggerStep(3, 'completed');
  return filtered;
}

function updateStatsDisplay() {
  document.getElementById('statChunks').textContent = state.chunks.length;
  document.getElementById('statVectors').textContent = state.chunks.length;
  document.getElementById('statWords').textContent = state.vocabulary.size;
}

// ==========================================================================
// RAG Pipeline Execution & Query Handling
// ==========================================================================
async function submitUserQuery() {
  const userInputEl = document.getElementById('userInput');
  const query = userInputEl.value.trim();
  if (!query) return;

  userInputEl.value = '';
  appendMessage('user', query);

  if (state.chunks.length === 0) {
    appendMessage('ai', '⚠️ İndekslenmiş herhangi bir belge bulunamadı! Lütfen önce sol panelden belge yükleyin veya "Örnek Belgeleri Yükle" butonuna basın.');
    return;
  }

  // Visual Pipeline Animation Progress
  resetPipelineVisualizer();
  document.getElementById('pipelineStatusText').textContent = 'İşleniyor...';

  // 1 & 2: Instant Tokenization & Retrieval
  triggerStep(1, 'completed');
  triggerStep(2, 'completed');

  // 3: Similarity Match
  const topMatches = retrieveTopK(query, state.config.topK, state.config.minScore);
  renderContextInspector(topMatches);

  // 4: Augment Prompt
  triggerStep(4, 'active');
  const augmentedPrompt = buildAugmentedPrompt(query, topMatches);
  document.getElementById('rawPromptView').textContent = augmentedPrompt;
  triggerStep(4, 'completed');

  // 5: Local Generation
  triggerStep(5, 'active');
  const aiMessageId = appendMessage('ai', '<span class="typing-dots"><span></span><span></span><span></span></span> <em>Yanıt oluşturuluyor, lütfen bekleyin...</em>', true);

  try {
    let responseText = '';
    if (state.config.mode === 'ollama' && state.ollamaAvailable) {
      responseText = await generateWithOllama(augmentedPrompt);
    } else {
      responseText = generateWithBuiltinRAG(query, topMatches);
    }

    updateMessageContent(aiMessageId, responseText, topMatches);
    triggerStep(5, 'completed');
    document.getElementById('pipelineStatusText').textContent = 'Tamamlandı';
  } catch (err) {
    console.error('LLM Üretim Hatası:', err);
    // Fallback to built-in generator
    const fallbackText = generateWithBuiltinRAG(query, topMatches);
    updateMessageContent(aiMessageId, fallbackText + `\n\n*(Not: Yerel Ollama yanıt vermediği için dahili çevrimdışı motor devreye girdi.)*`, topMatches);
    triggerStep(5, 'completed');
    document.getElementById('pipelineStatusText').textContent = 'Çevrimdışı Modda Tamamlandı';
  }
}

function buildAugmentedPrompt(query, matches) {
  let contextStr = '';
  matches.forEach((item, idx) => {
    contextStr += `[Belge ${idx + 1}: ${item.chunk.docName} (Skor: %${item.scorePercent})]\n${item.chunk.text}\n\n`;
  });

  return `Sistem Talimatı: Sen %100 yerel ve çevrimdışı çalışan bir RAG asistanısın. Aşağıda kullanıcının yüklediği belgelerden getirilen özel bağlam bilgisi verilmiştir. Sadece verilen bu bağlama dayanarak kullanıcının sorusunu Türkçe olarak detaylı ve açık bir şekilde yanıtla. Cümleleri ve kelimeleri hiçbir şekilde yarım bırakma.

BAGLAM BİLGİLERİ:
${contextStr || 'İlgili bağlam bulunamadı.'}

KULLANICI SORUSU:
${query}

YANIT:`;
}

function generateWithBuiltinRAG(query, matches) {
  if (matches.length === 0) {
    return `Yüklenen belgelerde "${query}" sorusuyla eşleşen yeterli bilgi bulunamadı. Lütfen RAG konfigürasyonundan **Min. Benzerlik Eşiği** değerini düşürmeyi (%10 veya %0 yapmayı) veya belgelerinize yeni konular eklemeyi deneyin.`;
  }

  // Smart extractive summary synthesis from top matched chunks
  let summary = `**"${query}"** hakkındaki belgelerinize dayanan yanıt:\n\n`;
  
  matches.forEach((m, idx) => {
    let snippet = m.chunk.text.replace(/\n+/g, ' ').trim();
    if (snippet.length > 350) {
      const lastSpace = snippet.lastIndexOf(' ', 350);
      snippet = (lastSpace > 200 ? snippet.substring(0, lastSpace) : snippet.substring(0, 350)) + '...';
    }
    summary += `▪ **${m.chunk.docName}** (Parça #${m.chunk.index} - %${m.scorePercent} uyum):\n> "${snippet}"\n\n`;
  });

  summary += `📌 **Özet Değerlendirme:** Belgelerdeki ilgili metin parçaları kelime bütünlüğü korunarak aktarılmıştır.`;
  return summary;
}

async function generateWithOllama(prompt) {
  const endpoint = `${state.config.endpointUrl}/api/generate`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000);

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        model: state.config.customModelName || 'llama3',
        prompt: prompt,
        stream: false,
        options: {
          temperature: state.config.temperature
        }
      })
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Ollama HTTP Hata: ${response.status}`);
    }

    const data = await response.json();
    return data.response;
  } catch (err) {
    clearTimeout(timeoutId);
    throw err;
  }
}

// ==========================================================================
// UI Rendering & Inspection Components
// ==========================================================================
function renderContextInspector(matches) {
  const container = document.getElementById('chunksContainer');
  const countBadge = document.getElementById('contextMatchCount');
  const emptyContext = document.getElementById('emptyContext');

  countBadge.textContent = `${matches.length} Parça`;

  if (matches.length === 0) {
    emptyContext.style.display = 'block';
    container.innerHTML = '';
    return;
  }

  emptyContext.style.display = 'none';
  container.innerHTML = matches.map((item, idx) => `
    <div class="chunk-card">
      <div class="chunk-card-header">
        <span><i class="fa-solid fa-file-text"></i> ${item.chunk.docName} (Parça #${item.chunk.index})</span>
        <span class="match-score">%${item.scorePercent} Uyum</span>
      </div>
      <div class="chunk-text">${escapeHTML(item.chunk.text)}</div>
    </div>
  `).join('');
}

function appendMessage(role, text, isPending = false) {
  const chatHistory = document.getElementById('chatHistory');
  const msgId = 'msg-' + Date.now();

  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}-message`;
  msgDiv.id = msgId;

  const avatarIcon = role === 'user' ? 'fa-user' : 'fa-robot';

  msgDiv.innerHTML = `
    <div class="message-avatar"><i class="fa-solid ${avatarIcon}"></i></div>
    <div class="message-content">
      <div class="msg-text">${formatMarkdown(text)}</div>
      <div class="citations-container"></div>
    </div>
  `;

  chatHistory.appendChild(msgDiv);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return msgId;
}

function updateMessageContent(msgId, text, matches) {
  const msgEl = document.getElementById(msgId);
  if (!msgEl) return;

  const msgTextEl = msgEl.querySelector('.msg-text');
  msgTextEl.innerHTML = formatMarkdown(text);

  // Render Citations Chip Accordion
  if (matches && matches.length > 0) {
    const citationsEl = msgEl.querySelector('.citations-container');
    citationsEl.innerHTML = `
      <div class="citations-box">
        <div class="citations-header"><i class="fa-solid fa-bookmark"></i> Kullanılan Yerel Kaynaklar:</div>
        <div class="chips-row">
          ${matches.map(m => `
            <span class="citation-chip" title="${escapeHTML(m.chunk.text.substring(0, 150))}">
              <i class="fa-solid fa-file-lines"></i> ${m.chunk.docName} <span class="match-score">%${m.scorePercent}</span>
            </span>
          `).join('')}
        </div>
      </div>
    `;
  }

  const chatHistory = document.getElementById('chatHistory');
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function formatMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/^> (.*$)/gm, visualBlockQuote)
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
}

function visualBlockQuote(match, p1) {
  return `<blockquote style="border-left: 3px solid var(--accent-cyan); padding-left: 10px; margin: 6px 0; color: var(--text-muted); font-style: italic;">${p1}</blockquote>`;
}

function escapeHTML(str) {
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}

// Visual Pipeline Indicator Animations
function resetPipelineVisualizer() {
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById(`step${i}`);
    el.className = 'step-box';
  }
}

function triggerStep(stepNum, statusClass) {
  const el = document.getElementById(`step${stepNum}`);
  if (el) {
    el.className = `step-box ${statusClass}`;
  }
}

// ==========================================================================
// Ollama Connection Health Check & Modal Settings
// ==========================================================================
async function checkOllamaHealth() {
  const badge = document.getElementById('connectionBadge');
  const textEl = document.getElementById('connectionText');

  try {
    const res = await fetch(`${state.config.endpointUrl}/api/tags`, { method: 'GET' });
    if (res.ok) {
      const data = await res.json();
      state.ollamaAvailable = true;
      state.availableModels = data.models || [];

      badge.className = 'status-badge online';
      textEl.textContent = `Ollama Bağlı (${state.availableModels.length} Model Bulundu)`;
      populateModelDropdown(state.availableModels);
      return;
    }
  } catch (err) {
    // Offline / Ollama not running
  }

  state.ollamaAvailable = false;
  badge.className = 'status-badge offline';
  textEl.textContent = 'Ollama Servisi Pasif (Dahili Engine Aktif)';
}

function populateModelDropdown(models) {
  const selector = document.getElementById('modelSelector');
  if (!models || models.length === 0) return;

  // Keep builtin option
  const optionsHtml = models.map(m => `<option value="${m.name}">${m.name} (${formatBytes(m.size)})</option>`).join('');
  selector.innerHTML = optionsHtml + `<option value="builtin-offline">⚡ Dahili Çevrimdışı Engine</option>`;
  
  if (models.length > 0) {
    selector.value = models[0].name;
    state.config.selectedModel = models[0].name;
    state.config.customModelName = models[0].name;
    state.config.mode = 'ollama';
    updateActiveModelFooter();
  }
}

function updateActiveModelFooter() {
  const footerEl = document.getElementById('activeModelFooter');
  if (state.config.mode === 'ollama') {
    footerEl.textContent = `Aktif Model: ${state.config.customModelName} (Ollama Endpoint)`;
  } else {
    footerEl.textContent = `Aktif Model: Dahili Çevrimdışı Engine`;
  }
}

function openSettingsModal() {
  document.getElementById('settingsModal').classList.add('active');
}

function closeSettingsModal() {
  document.getElementById('settingsModal').classList.remove('active');
}

async function testConnectionInModal() {
  const url = document.getElementById('endpointUrl').value.trim();
  const resEl = document.getElementById('testResultText');
  resEl.textContent = 'Test ediliyor...';
  resEl.style.color = 'var(--text-muted)';

  try {
    const res = await fetch(`${url}/api/tags`);
    if (res.ok) {
      const data = await res.json();
      resEl.textContent = `✅ Bağlantı Başarılı! (${(data.models || []).length} model bulundu)`;
      resEl.style.color = 'var(--accent-emerald)';
    } else {
      resEl.textContent = `⚠️ HTTP Yanıt: ${res.status}`;
      resEl.style.color = 'var(--accent-amber)';
    }
  } catch (err) {
    resEl.textContent = '❌ Bağlanılamadı (Ollama kapalı veya CORS engeli)';
    resEl.style.color = 'var(--accent-coral)';
  }
}

function saveSettings() {
  state.config.endpointUrl = document.getElementById('endpointUrl').value.trim();
  state.config.customModelName = document.getElementById('customModelName').value.trim();
  
  const modeVal = document.querySelector('input[name="mode"]:checked').value;
  state.config.mode = modeVal;

  checkOllamaHealth();
  updateActiveModelFooter();
  closeSettingsModal();
}
