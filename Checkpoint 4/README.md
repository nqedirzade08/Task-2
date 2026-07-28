# RAG Pipeline — Checkpoint 4: Retrieval + Prompt Qurulması
 
Checkpoint 3-də tapılan chunk-lar indi **aydın struktura malik bir prompt-a** inteqrasiya olunur: təlimatlar (system), tapılan kontekst və istifadəçi sualı bir-birindən dəqiq ayrılır. Aşağı etibarlılıqlı retrieval halları üçün əlavə xəbərdarlıq mexanizmi də var.
 
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
├── prompt_builder.py                                          — prompt qurulması (YENİ)
├── generation.py                                                — LLM cavab generasiyası (YENİ)
├── main.py                                                       — CLI: tam axın nümayişi
├── test_chunk_boundary.py, documents/                             — əvvəlki checkpoint-lərdən
├── .env / .env.example, .gitignore, requirements.txt
```

## Nümunə işə salma nəticəsi
 
**Sual 1 (chunk sərhədi faktı) — düzgün cavablandı:**
```
SUAL: ADNSU-da Kompüter Mühəndisliyi ixtisası üzrə 2025-ci il qəbul balı neçə idi?
 
[KONTEKST-də 3 chunk, ən yaxını məsafə=0.4068 — aşağı etibarlılıq siqnalı YOXDUR]
 
--- MODEL CAVABI ---
ADNSU-da Kompüter Mühəndisliyi ixtisası üzrə 2025-ci il qəbul imtahanında ən aşağı
keçid balı 342.7 bal olub. (Mənbə: adnsu_melumati.txt)
```
 
**Sual 2 (hallüsinasiya test sualı) — uğurla rədd edildi:**
```
SUAL: ADNSU-da xarici tələbələr üçün illik təhsil haqqı neçə manatdır?
 
[KONTEKST-də 3 chunk, ən yaxını məsafə=0.5311 — 0.50 həddini aşır]
 
[SİSTEM QEYDİ: Ən yaxın tapılan parçanın oxşarlıq məsafəsi (0.5311) 0.5 həddini
aşır — bu, sualın KONTEKST-də birbaşa cavablandırılmaya bilməyəcəyini göstərir.
Bu halda uydurma cavab vermə, məlumatın olmadığını bildir.]
 
--- MODEL CAVABI ---
Bu suala cavab verə bilmirəm, çünki mövcud sənədlərdə ADNSU-da xarici tələbələr
üçün illik təhsil haqqı barədə heç bir məlumat yoxdur.
```
 
