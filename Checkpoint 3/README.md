# RAG Pipeline — Checkpoint 3: Vektor Saxlama + Oxşarlıq Axtarışı
 
Checkpoint 2-də generasiya olunan chunk embedding-ləri **Chroma** vektor verilənlər bazasına yazılır və sual embedding-i ilə oxşarlıq (cosine similarity) axtarışı aparılır.
 
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
├── ingest.py, chunking.py, embeddings.py   — əvvəlki checkpoint-lərdən
├── vectorstore.py                             — Chroma inteqrasiyası (YENİ)
├── main.py                                     — CLI: ingest → chunk → embed → saxla → axtarış testləri
├── test_chunk_boundary.py                       — Checkpoint 1-dən
├── documents/                                     — nümunə sənədlər
├── .env / .env.example, .gitignore, requirements.txt
```
## Nümunə işə salma nəticəsi
 
```
=== RAG Checkpoint 3 — Vektor Saxlama + Oxşarlıq Axtarışı ===
Yüklənmiş sənəd sayı: 2
  - adnsu_melumati.txt (1791 simvol)
  - bmu_melumati.txt (992 simvol)
Chunking parametrləri: chunk_size=500, chunk_overlap=100
Ümumi chunk sayı: 8
Embedding modeli: nvidia/nv-embedqa-e5-v5
[Embedding] 8/8 chunk emb edildi
Vektor bazasına 8 chunk yazıldı (Chroma, persist qovluğu: chroma_db/)
 
=== Oxşarlıq Axtarışı Testləri ===
 
Sual: ADNSU-da Kompüter Mühəndisliyi ixtisası üzrə 2025-ci il qəbul balı neçə idi?
  1) [adnsu_melumati.txt::chunk_1] məsafə=0.4068
     ...Kompüter Mühəndisliyi ixtisası üzrə 2025-ci il qəbul imtahanında ən aşağı
     keçid balı 342.7 bal olub, bu da əvvəlki ilə nisbətən 15 bal yüksəkdi...
  2) [adnsu_melumati.txt::chunk_0] məsafə=0.4132
  3) [bmu_melumati.txt::chunk_1] məsafə=0.4616
 
Sual: ADNSU-da xarici tələbələr üçün illik təhsil haqqı neçə manatdır?
  1) [adnsu_melumati.txt::chunk_0] məsafə=0.5311
  2) [adnsu_melumati.txt::chunk_4] məsafə=0.5659
  3) [adnsu_melumati.txt::chunk_3] məsafə=0.5680
 
Sual: BMU nə vaxt yaradılıb və hansı ixtisasları var?
  1) [bmu_melumati.txt::chunk_0] məsafə=0.4945
     Bakı Mühəndislik Universiteti (BMU) 2016-cı ildə yaradılmış, müasir
     yanaşmaları ilə seçilən gənc dövlət universitetidir...
  2) [bmu_melumati.txt::chunk_1] məsafə=0.5177
  3) [adnsu_melumati.txt::chunk_1] məsafə=0.5771
```
 
### Nəticələrin təhlili
 
**Sual 1 (chunk sərhədi faktı) — uğurlu:** Ən yaxın nəticə (məsafə=0.4068) məhz "342.7 bal" faktını ehtiva edən `chunk_1`-dir. Checkpoint 1-də overlap strategiyası sayəsində bu fakt bütöv qaldığı üçün, embedding də onu tam və dəqiq təmsil edir, retrieval onu birinci sırada tapır.
 
**Sual 2 (hallüsinasiya test sualı) — gözlənilən nəticə:** Diqqət edin ki, bu sualın ən yaxın nəticəsinin məsafəsi (0.5311) sual 1-in ən yaxın nəticəsinin məsafəsindən (0.4068) **nəzərəçarpacaq dərəcədə yüksəkdir**. Bu məntiqlidir — sənədlərdə təhsil haqqı barədə heç nə olmadığı üçün, ən "yaxın" chunk-lar belə əslində sualı cavablandırmır, sadəcə mövzu baxımından bir qədər yaxındır (universitet haqqında ümumi məlumat). Bu məsafə fərqi **növbəti checkpoint-də** (generation) LLM-ə "bu chunk-lar sualı əslində cavablandırmır, uydurma etmə" siqnalı kimi istifadə oluna bilər (məs. məsafə həddi/threshold ilə).
 
**Sual 3 (çox-sənədli seçim) — uğurlu:** Sual yalnız BMU haqqında olduğu üçün, ən yaxın 2 nəticə düzgün olaraq `bmu_melumati.txt`-dən gəlir, ADNSU-ya aid chunk yalnız 3-cü (ən uzaq) sırada görünür.
