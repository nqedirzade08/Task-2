# RAG Pipeline — Checkpoint 5: Mənbə İstinadı ilə Cavab Generasiyası
 
Checkpoint 4-də cavab modeldən sərbəst mətn şəklində, mənbəni yalnız prosada ("(Mənbə: ...)" formasında) qeyd edirdi. Bu checkpoint-də mənbə istinadı **strukturlaşdırılmış və yoxlanıla bilən** formaya keçirilir — hər cavab hansı konkret chunk-lardan (fayl + chunk indeksi) istifadə etdiyini aydın göstərir.
 
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
 
3. `.env` faylı artıq real API key ilə doludur.
4. Pipeline-ı işə salın:
```
   python main.py
```
 
## Fayl strukturu
 
```
├── ingest.py, chunking.py, embeddings.py, vectorstore.py   — əvvəlki checkpoint-lərdən
├── prompt_builder.py                                          — etiketli kontekst + JSON tələbi (YENİLƏNİB)
├── citation.py                                                  — JSON parsing/validasiya/istinad həlli (YENİ)
├── generation.py                                                  — LLM çağırışı + JSON retry (YENİLƏNİB)
├── main.py                                                         — CLI: tam axın nümayişi
├── test_chunk_boundary.py, documents/                               — əvvəlki checkpoint-lərdən
├── .env / .env.example, .gitignore, requirements.txt
```
## Nümunə işə salma nəticəsi
 
**Sual 1 (chunk sərhədi faktı):**
```
--- MƏNBƏ İSTİNADLI CAVAB ---
ADNSU-da Kompüter Mühəndisliyi ixtisası üzrə 2025-ci il qəbul imtahanında ən aşağı
keçid balı 342.7 bal olub.
 
Mənbələr:
  - adnsu_melumati.txt (chunk 1, məsafə=0.4068)
  - adnsu_melumati.txt (chunk 0, məsafə=0.4132)
```
Model faktın **iki** overlap-lı chunk-da da mövcud olduğunu düzgün tutub, hər ikisini mənbə kimi göstərib.
 
**Sual 2 (hallüsinasiya test sualı):**
```
(Aşağı etibarlılıq siqnalı aktivdir — ən yaxın məsafə həddi 0.5-dən yüksəkdir)
 
--- MƏNBƏ İSTİNADLI CAVAB ---
Bu suala cavab verə bilmirəm, çünki mövcud sənədlərdə bu barədə məlumat yoxdur.
```
Heç bir mənbə göstərilməyib (`sources: []`) — model uydurmayıb.
 
**Sual 3 (BMU tarixi) — maraqlı sərhəd halı:**
```
(Aşağı etibarlılıq siqnalı aktivdir — ən yaxın məsafə həddi 0.5-dən yüksəkdir)
 
--- MƏNBƏ İSTİNADLI CAVAB ---
Bakı Mühəndislik Universiteti (BMU) 2016-cı ildə yaradılmışdır.
 
Mənbələr:
  - bmu_melumati.txt (chunk 0, məsafə=0.5598)
```
 
