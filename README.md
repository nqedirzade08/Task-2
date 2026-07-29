# Task 2 — RAG Pipeline: "Sənədlərinlə Danış"

## Layihənin izahı

Bu layihə istifadəçinin bir sənəd toplusu (bu layihədə: universitetlər/ixtisaslar haqqında `.txt` sənədlər, `.pdf` dəstəyi də daxildir) üzərində sual verə biləcəyi və cavabın **yalnız həmin sənədlərin məzmununa əsaslandığı** bir RAG (Retrieval-Augmented Generation) pipeline-ıdır. Sistem embedding + vektor verilənlər bazası + LLM birləşməsi ilə işləyir: sual verildikdə, ən uyğun sənəd parçaları (chunk) tapılır, bunlar LLM-ə aydın strukturlaşdırılmış prompt daxilində ötürülür, LLM isə cavabı yalnız bu kontekstə əsaslanaraq, mənbə istinadı ilə generasiya edir.

Layihənin iki əsas mühəndislik çətinliyi xüsusi diqqətlə həll olunub:
1. **Chunk sərhədi problemi** — vacib faktların iki parça arasında bölünməsinin qarşısını almaq (overlap strategiyası)
2. **Hallüsinasiya problemi** — sənədlərdə olmayan sual verildikdə modelin rəqəm/fakt uydurmasının qarşısını almaq

Hər ikisi qəsdən qurulmuş test ssenariləri ilə yoxlanılıb və uğurla keçib (aşağıda ətraflı).

## Texnologiya seçimi

- **Dil:** Python
- **LLM provayder:** NVIDIA API (OpenAI-uyğun endpoint, `integrate.api.nvidia.com`)
  - **Chat/generasiya modeli:** `deepseek-ai/deepseek-v4-flash` (əvvəlcə `deepseek-v4-pro` sınanıb, lakin davamlı timeout verdiyi üçün flash versiyasına keçilib)
  - **Embedding modeli:** `nvidia/nv-embedqa-e5-v5` (1024 ölçülü vektor; ilkin seçim `baai/bge-m3` idi, amma hesabda davamlı server xətası verdiyi üçün dəyişdirildi)
- **Vektor verilənlər bazası:** Chroma (`PersistentClient`, cosine məsafə metrikası) — server tələb etmir, layihəni sınamaq üçün əlavə hesab lazım deyil
- **Kitabxanalar:** `openai`, `python-dotenv`, `chromadb`, `pypdf`

## Checkpoint-lər üzrə inkişaf

### Checkpoint 1 — Sənəd Ingestion + Chunking (20 bal)
`.txt`/`.pdf` sənədləri yüklənir, `chunk_size=500`/`chunk_overlap=100` sliding-window strategiyası ilə bölünür. **Chunk sərhədi trick testi:** qəsdən qurulmuş sənəddə açar fakt ("342.7 bal") naiv overlap-sız bölmədə hərfiyyən ortadan kəsilir, overlap-lı versiyada isə tam saxlanılır — real test nəticələri ilə sübut olunub.

### Checkpoint 2 — Chunk-lar üçün Embedding Generasiyası (15 bal)
Hər chunk `nvidia/nv-embedqa-e5-v5` modeli ilə 1024 ölçülü vektora çevrilir, batch-larla (16/sorğu) və retry məntiqi ilə. Nəticə (chunk + vektor + metadata) sonrakı mərhələ üçün saxlanılır.

### Checkpoint 3 — Vektor Saxlama + Oxşarlıq Axtarışı (20 bal)
Embedding-lər Chroma-ya yazılır, sual üçün oxşarlıq (cosine) axtarışı aparılır. Müşahidə: real cavabı olan suallarda ən yaxın məsafə ~0.40-larda, cavabı olmayan sualda isə ~0.53+ — bu fərq sonrakı checkpoint-lərdə hallüsinasiya müdafiəsinin əsasını təşkil edir.

### Checkpoint 4 — Retrieval + Prompt Qurulması (20 bal)
Tapılan chunk-lar aydın strukturlaşdırılmış prompt-a inteqrasiya olunur: təlimatlar ayrıca `system` mesajında, kontekst və sual isə `[KONTEKST]`/`[SUAL]` etiketləri ilə `user` mesajında aydın ayrılır. Checkpoint 3-dəki məsafə müşahidəsi əsasında `SIMILARITY_DISTANCE_THRESHOLD=0.50` aşıldıqda prompt-a avtomatik xəbərdarlıq əlavə olunur. **Nəticə: hallüsinasiya test sualında model uydurma etmədən "məlumat yoxdur" cavabı verdi** — layihənin əsas fokus testi uğurla keçdi.

### Checkpoint 5 — Mənbə İstinadı ilə Cavab Generasiyası (15 bal)
Cavab strukturlaşdırılmış JSON formatına keçirildi: hər chunk `[C1]`, `[C2]` kimi etiketlənir, model istifadə etdiyi etiketləri `"sources"` sahəsində qaytarır, bunlar isə real fayl adı/chunk nömrəsinə həll olunur. Əlavə qorunma: modelin mövcud olmayan etiketə istinad etməsi (öz sitatını uydurması) aşkarlanır.

### Checkpoint 6 — "Sənədlərdə Yoxdur" Halının İdarə Olunması (10 bal)
"Tapılmadı" halı strukturlaşdırılmış `answer_found_in_context: true/false` sahəsinə çevrildi, məntiqi ziddiyyət yoxlaması əlavə olundu (false + boş olmayan mənbə siyahısı qəbul edilmir). **6 test halından ibarət sistematik qiymətləndirmə** (3 cavabı olan, 3 olmayan, o cümlədən "mövzu uyğundur amma konkret rəqəm yoxdur" kimi çətin bir hal) — **6/6 (100%) doğruluq**.

## İki əsas keyfiyyət testi — yekun status

| Test | Nəticə |
|---|---|
| **Chunk sərhədi trick-i** | ✅ Keçdi (Checkpoint 1) — overlap olmadan fakt bölünür, overlap ilə tam qalır |
| **Hallüsinasiya trick-i** | ✅ Keçdi (Checkpoint 4, təsdiqləndi Checkpoint 6-da 6/6 sistematik testlə) |

## Ümumi təhlükəsizlik təcrübəsi

Task 1-dəki eyni prinsiplər qorunub:
- API key yalnız `.env` faylındadır, kodda heç vaxt açıq yazılmayıb
- Hər checkpoint qovluğunda `.gitignore` `.env`-i (və Chroma-nın yerli data qovluğunu, generasiya olunan JSON/pickle fayllarını) istisna edir
- Hər checkpoint-in `.env.example` faylında yalnız placeholder dəyər var

## Qovluq strukturu (bu repository)

```
Task 2/
├── Checkpoint 1/    — Sənəd ingestion + chunking
├── Checkpoint 2/    — Embedding generasiyası
├── Checkpoint 3/    — Vektor saxlama + oxşarlıq axtarışı
├── Checkpoint 4/    — Retrieval + prompt qurulması
├── Checkpoint 5/    — Mənbə istinadı ilə cavab generasiyası
└── Checkpoint 6/    — "Sənədlərdə yoxdur" halının idarə olunması
```

Hər checkpoint qovluğu tam işlək, müstəqil işə salına bilən layihə saxlayır və əvvəlki checkpoint-in bütün fayllarını özündə daşıyır (kumulyativ inkişaf) — yəni, məsələn, Checkpoint 6-nı işə salmaq bütün pipeline-ı (ingestion-dan cavab generasiyasına qədər) əhatə edir.

## İşə salma (istənilən checkpoint qovluğu üçün eyni)

```bash
cd "Checkpoint N"
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py
```

(`.env` faylı artıq real API key ilə doludur; öz key-inizlə işləmək istəsəniz `.env.example`-i əsas götürün.)

Ətraflı izah, arxitektur qərarları və real test log-ları hər checkpoint-in öz `README.md` faylındadır.
