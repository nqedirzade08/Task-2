# RAG Pipeline — Checkpoint 6: "Sənədlərdə Yoxdur" Halının İdarə Olunması
 
Əvvəlki checkpoint-lərdə "sənədlərdə yoxdur" davranışı yalnız prosada (cavab mətnində) və qeyri-formal şəkildə yoxlanılırdı. Bu checkpoint-də bu, **strukturlaşdırılmış, ölçülə bilən bir sahəyə** (`answer_found_in_context: true/false`) çevrilir və **6 test halından ibarət sistematik qiymətləndirmə** ilə sübut olunur (3 cavabı olan, 3 cavabı olmayan sual).
 
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
4. Qiymətləndirməni işə salın:
```
   python main.py
```
 
## Fayl strukturu
 
```
├── ingest.py, chunking.py, embeddings.py, vectorstore.py   — əvvəlki checkpoint-lərdən
├── prompt_builder.py                                          — answer_found_in_context tələbi (YENİLƏNİB)
├── citation.py                                                  — validasiya + uyğunluq yoxlaması (YENİLƏNİB)
├── generation.py                                                  — əvvəlki checkpoint-dən
├── main.py                                                         — 6 test halı ilə qiymətləndirmə (YENİLƏNİB)
├── test_chunk_boundary.py, documents/                               — əvvəlki checkpoint-lərdən
├── .env / .env.example, .gitignore, requirements.txt
```
 
