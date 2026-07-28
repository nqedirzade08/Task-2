# RAG Pipeline — Checkpoint 2: Chunk-lar üçün Embedding Generasiyası
 
Checkpoint 1-də hazırlanan chunk-lar üçün NVIDIA API vasitəsilə vektor embedding-lər generasiya olunur — bu, sonrakı checkpoint-də (vektor verilənlər bazası + retrieval) axtarışın əsasını təşkil edəcək.
 
## Qurulum
 
1. Virtual mühit yaradın və aktivləşdirin:
```
   python -m venv venv
   venv\Scripts\activate
```
   Mac/Linux: `source venv/bin/activate`
 
2. Asılılıqları quraşdırın:
```
   pip install -r requirements.txt
```
 
3. `.env` faylı artıq real API key ilə doludur. Öz key-inizlə işləmək istəsəniz, `.env.example`-i `.env` adı ilə kopyalayıb özününküni yazın.
4. Pipeline-ı işə salın:
```
   python main.py
```
 
## Fayl strukturu
 
```
├── ingest.py                          — sənəd yükləmə (Checkpoint 1-dən)
├── chunking.py                          — chunking (Checkpoint 1-dən)
├── embeddings.py                          — embedding generasiyası (YENİ)
├── main.py                                 — CLI: ingest → chunk → embed → saxla
├── test_chunk_boundary.py                   — chunk sərhədi testi (Checkpoint 1-dən)
├── documents/                                 — nümunə sənədlər (Checkpoint 1-dən)
├── .env / .env.example
├── .gitignore
└── requirements.txt
```
## Nümunə işə salma nəticəsi
 
```
=== RAG Checkpoint 2 — Chunk-lar üçün Embedding Generasiyası ===
Yüklənmiş sənəd sayı: 2
  - adnsu_melumati.txt (1791 simvol)
  - bmu_melumati.txt (992 simvol)
Chunking parametrləri: chunk_size=500, chunk_overlap=100
Ümumi chunk sayı: 8
Embedding modeli: nvidia/nv-embedqa-e5-v5
[Embedding] 8/8 chunk emb edildi
Tam nəticə (vektorlarla birlikdə) 'chunks_with_embeddings.pkl' faylına yazıldı.
Oxunaqlı önizləmə 'chunks_with_embeddings_preview.json' faylına yazıldı (vektorun ilk 5 ölçüsü ilə).
Hər chunk üçün vektor ölçüsü: 1024
Cəmi embedding edilmiş chunk: 8
```
 
