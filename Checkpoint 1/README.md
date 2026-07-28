# RAG Pipeline — Checkpoint 1: Sənəd Ingestion + Chunking
 
"Sənədlərinlə Danış" RAG pipeline-ının ilk mərhələsi: istifadəçinin verdiyi sənədlər toplusunu (`.txt`, `.pdf`) yükləyib, sonrakı embedding/retrieval mərhələləri üçün məntiqli ölçü və overlap strategiyası ilə chunk-lara bölmək.
 
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
 
3. Pipeline-ı işə salın:
```
   python main.py
```
   `documents/` qovluğundakı bütün `.txt`/`.pdf` faylları yüklənir, chunk-lanır və nəticə `chunks_output.json`-a yazılır.
 
4. Chunk sərhədi/overlap testini işə salmaq üçün:
```
   python test_chunk_boundary.py
```
 
## Fayl strukturu
 
```
├── ingest.py                 — sənəd yükləmə (.txt, .pdf)
├── chunking.py                 — naiv (baseline) və overlap-lı (production) chunking funksiyaları
├── main.py                      — CLI: ingestion + chunking pipeline-ı işə salır
├── test_chunk_boundary.py        — chunk sərhədi/overlap demo testi
├── documents/
│   ├── adnsu_melumati.txt         — qəsdən chunk sərhədini sınayan test sənədi
│   └── bmu_melumati.txt            — əlavə nümunə sənəd (çox-sənədli ingestion üçün)
└── requirements.txt
```
## Çox-sənədli ingestion nümunəsi
 
**`main.py` real işə salma nəticəsi:**
 
```
=== RAG Checkpoint 1 — Sənəd Ingestion + Chunking ===
 
Yüklənmiş sənəd sayı: 2
  - adnsu_melumati.txt (1791 simvol)
  - bmu_melumati.txt (992 simvol)
 
Chunking parametrləri: chunk_size=500, chunk_overlap=100
 
Ümumi chunk sayı: 8
 
  - adnsu_melumati.txt: 5 chunk
  - bmu_melumati.txt: 3 chunk
 
Chunk-lar 'chunks_output.json' faylına yazıldı.
```
 
`.pdf` dəstəyi də `ingest.py`-da hazırdır (`pypdf` ilə) — `documents/` qovluğuna PDF faylı əlavə etsəniz, avtomatik oxunub eyni chunking axınından keçəcək.
