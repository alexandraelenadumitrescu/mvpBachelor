# Capitolul 3. Proiectarea arhitecturii soluției

---

## 3.1 Arhitectura generală a soluției

Soluția dezvoltată urmează o arhitectură client-server clasică, în care responsabilitățile sunt distribuite între o aplicație mobilă Android și un server de backend implementat în Python cu framework-ul FastAPI. Această separare a fost aleasă în mod deliberat pentru a valorifica puterea de calcul disponibilă pe server pentru operațiunile de inferență cu modele de mari dimensiuni, păstrând în același timp pe dispozitivul mobil procesele care beneficiază de latență redusă sau de accesul direct la hardware-ul camerei.

Clientul Android, scris exclusiv în Java, îndeplinește trei roluri distincte: interfața cu utilizatorul, inferența locală cu modele TensorFlow Lite optimizate și comunicarea cu serverul prin apeluri REST autentificate. Prin execuția locală a modelelor CLIP ViT-B/32 și a capului de detecție defecte bazat pe MobileNetV3Small, aplicația evită transmiterea imaginilor de dimensiuni mari prin rețea pentru cazul de utilizare principal — corecția de culoare — reducând semnificativ latența percepută de utilizator și consumul de date mobile.

Serverul, pornit cu Uvicorn pe portul 8000, expune peste treizeci de endpoint-uri REST organizate pe domenii funcționale: autentificare (prefixul `/auth`), detecție și anonimizare regiuni sensibile (`/blur-*`), căutare vectorială și corecție imagine (`/search*`, `/process`, `/batch`), clustering (`/cluster`), livrare bazată pe recunoaștere facială (`/delivery/*`), scraping web cu detecție riscuri (`/scrape/*`) și servicire de resurse statice (`/image/*`, `/lut/*`). Autentificarea este implementată prin tokens JWT cu algoritmul HS256 și expirare la 24 de ore, validate printr-un interceptor OkHttp pe partea de client și prin dependența FastAPI `get_current_active_user` pe server.

Stocarea datelor este eterogenă prin natura aplicației. Pe dispozitivul mobil, biblioteca Room (strat de abstractizare peste SQLite) gestionează entitatea `FavoritePhoto`. Pe server, un model SQLAlchemy peste SQLite stochează utilizatorii înregistrați. Vectorii hibrid pre-calculați pentru cele 3.499 de imagini de referință din setul MIT FiveK sunt persistați în formatul NumPy comprimat (`hybrid_vectors.npz`) și încărcați la pornire într-un index FAISS `IndexFlatIP` rezident în memorie, oferind căutare prin produs intern în timp subliniar.

Comunicarea dintre client și server se face exclusiv prin HTTP simplu (cleartext), justificat de faptul că aplicația operează în rețele locale private, telefon și server aflându-se în aceeași rețea Wi-Fi. Adresa IP a serverului este configurabilă de utilizator din interfața de login și persistată în SharedPreferences.

---

### Descrierea diagramei — Arhitectura generală a sistemului

**Scopul diagramei:** să ofere o vedere de ansamblu a tuturor componentelor sistemului, a tehnologiilor folosite și a fluxurilor principale de date dintre ele. Este diagrama de referință pentru orice cititor care vrea să înțeleagă structura macro a soluției înainte de a intra în detalii.

**De ce este necesară:** capitolul de proiectare al oricărei lucrări de licență trebuie să conțină o diagramă arhitecturală care să demonstreze că studentul a înțeles și a proiectat în mod deliberat structura sistemului, nu a construit ad-hoc.

**Tipul diagramei:** diagramă arhitecturală de componente (Component Diagram UML 2.x sau diagramă de blocuri arhitecturale).

**Nivelul de granularitate recomandat:** mediu — componente principale vizibile, fără a detalia metodele sau atributele interne.

**Elemente care trebuie reprezentate:**

*Blocul Android Client:*
- Subcomponenta UI/Activities (20+ ecrane)
- Subcomponenta Local ML: CLIP TFLite (512-dim), Defect Head TFLite (5-dim), FaceNet TFLite (128-dim)
- Subcomponenta Networking: ApiClient (Retrofit + OkHttp), interceptor JWT
- Subcomponenta Storage: Room Database (FavoritePhoto)

*Blocul FastAPI Server (:8000):*
- Router Auth (`auth/router.py`)
- Router Scrape (`scrape_api/router.py`)
- Handler Blur (`blur_api/`)
- Handler Delivery (`photo_mailer/`)
- Handler Search/Process (inline în `server.py`)
- FAISS Index (3.499 vectori 517-dim, rezident în memorie)
- YOLOv8n COCO (ecrane, documente)
- badge_model.pt fine-tuned (Cards, Lanyard)
- CLIP ViT-B/32 via OpenCLIP (pentru căutarea server-side opțională)
- MobileNetV3Small Defect Head (`defect_head.pt`)
- SQLite Users DB (SQLAlchemy)
- Fișiere statice: imagini FiveK RAW/editate, LUT cache

*Fluxuri de date:*
- Android → Server: REST HTTP, multipart/form-data sau JSON, cu header `Authorization: Bearer {token}`
- Android → Server: bytes imagine (pentru anonimizare, delivery, scraper)
- Server → Android: JSON (regiuni, scoruri, vectori) sau bytes JPEG (imagine blurată)
- Android ↔ Local TFLite: tensor float32 (inferență locală, fără rețea)

**Ce trebuie evidențiat vizual:** granița client/server, modelele ML care rulează local față de cele care rulează pe server, faptul că vectorizarea pentru corecția de culoare se face local.

**Informații care nu trebuie omise:** modelul badge_model.pt este fine-tuned (nu general COCO), FAISS rulează in-memory, comunicarea este HTTP nu HTTPS.

**Contribuția la înțelegerea arhitecturii:** permite cititorului să înțeleagă în 30 de secunde unde rulează fiecare componenta și de ce a fost luată această decizie de distribuție a calculului.

**Specificație pentru realizarea diagramei:**
Desenați două blocuri principale separate vizual: „Android Client" și „FastAPI Server :8000". În interiorul blocului Android, creați trei subblocuri: „Local Inference" (conținând trei dreptunghiuri etichetate CLIP TFLite / 512-dim, Defect Head TFLite / 5-dim, FaceNet TFLite / 128-dim), „Networking" (ApiClient, OkHttp Interceptor JWT), „Persistence" (Room SQLite — FavoritePhoto). În interiorul blocului Server, creați subgrupuri pentru: Routers (Auth, Scrape, Blur, Delivery, Search), Models (YOLOv8n COCO, badge_model.pt, CLIP ViT-B/32, MobileNetV3 Defect Head), Storage (FAISS IndexFlatIP in-memory, SQLite Users, Filesystem FiveK + LUT cache). Conectați cele două blocuri principale cu o săgeată bidirecțională etichetată „HTTP REST / JWT". Adăugați o săgeată internă în blocul Android de la UI/Activities spre Local Inference etichetată „TFLite inference". Adăugați o săgeată de la Local Inference spre Networking etichetată „vector 517-dim". Folosiți culori diferite pentru client și server.

---

## 3.2 Arhitectura aplicației

### 3.2.1 Arhitectura aplicației Android

Aplicația Android este structurată în jurul unei ierarhii de clase de bază care implementează pattern-ul Template Method, eliminând duplicarea codului pentru operațiunile comune tuturor ecranelor. Clasa abstractă `BaseActivity`, care extinde `AppCompatActivity`, introduce trei responsabilități transversale: verificarea autentificării în metoda `onStart()` cu redirecționare automată spre `LoginActivity` dacă utilizatorul nu este autentificat, accesul la serviciul Retrofit prin metoda `api()` care delegă la singleton-ul `ApiClient`, și afișarea erorilor prin `showError()`.

`BaseLocalActivity` extinde `BaseActivity` și adaugă gestionarea unui `ExecutorService` cu un singur fir de execuție, creat în `onCreate()` și închis în `onDestroy()`. Această clasă introduce convenția că referința `primaryButton` și `progressBar` din fiecare activitate derivată trebuie asignate în `onCreate()`, permițând metodei moștenite `setProcessing(boolean)` să controleze automat starea de vizualizare a acestor componente fără cod suplimentar în subclase. `BaseServerActivity` extinde la rândul său `BaseLocalActivity` fără a adăuga logică proprie, servind ca marcare semantică pentru activitățile care efectuează apeluri de rețea.

Toate operațiunile asincrone din aplicație urmează același pattern: lucrul în fundal se execută prin `executor.execute(() -> { ... })`, iar actualizările UI se fac prin `runOnUiThread(() -> { ... })`. Aplicația nu folosește ViewModel, LiveData sau Coroutines — decizia de a rămâne la Java pur și la ExecutorService a fost motivată de portabilitate și de claritatea fluxului de date.

Comunicarea cu serverul este centralizată în `ApiClient`, implementat ca singleton cu dublă verificare a inițializării. La prima pornire a aplicației, `PhotoMatchApp.onCreate()` apelează `ApiClient.init(context)`, care citește IP-ul serverului din SharedPreferences (cheia `server_ip`, valoare implicită `10.33.163.180`) și construiește instanța Retrofit cu un client OkHttp configurat cu timeout de 600 de secunde pentru citire (necesar pentru descărcarea loturilor mari de imagini) și cu un interceptor care injectează header-ul `Authorization: Bearer {token}` pe toate request-urile cu excepția endpoint-urilor de autentificare.

Interfața `ApiService`, adnotată cu Retrofit, definește toate cele douăzeci și unu de metode de comunicare cu serverul. Transferul imaginilor se face în două moduri: pentru endpoint-urile bazate pe vectori (căutare, clustering) se trimite un body JSON cu array-uri float, iar pentru endpoint-urile care necesită analiza pixelilor (anonimizare, delivery, scraper) se folosesc request-uri multipart cu câmpul `file` de tip `image/jpeg`.

Persistența locală este gestionată de biblioteca Room cu o singură entitate, `FavoritePhoto`, care stochează base64-uri ale imaginilor originale și corectate, URI-ul sursei, numele imaginii de referință recuperate și un câmp JSON pentru îmbunătățirile detectate. Baza de date este un singleton accesat prin `AppDatabase.get(context)` și este utilizată exclusiv din `ResultsActivity` și `FavoritesActivity`.

### 3.2.2 Arhitectura serverului FastAPI

Serverul este organizat în module cu responsabilități clare. Fișierul principal `server.py` încarcă configurația din `.env` prin `python-dotenv`, inițializează modelele ML (CLIP via OpenCLIP, MobileNetV3Small defect head), construiește indexul FAISS din vectorii pre-calculați și definește inline endpoint-urile pentru operațiunile de căutare, corecție, clustering și servire de resurse statice. Endpoint-urile specializate sunt delegate către routere separate montate cu prefix: `auth/router.py` pentru autentificare și `scrape_api/router.py` pentru scraping.

Modulul `blur_api/` grupează logica de anonimizare: `local_detector.py` orchestrează detecția locală combinând YOLOv8n COCO pentru ecrane și documente cu `detect_badges()` din `badge_detector.py` pentru carduri de conferință; `gemini.py` implementează alternativa cloud prin API-ul Google Gemini 2.0 Flash; `blur.py` aplică filtrul Gaussian cu kernel 51×51 pe regiunile identificate. Modulul `photo_mailer/` conține logica de livrare: scraping pagini de angajați, construirea bazei de embeddings faciale, matching și trimitere SMTP.

---

## 3.3 Diagrame de cazuri de utilizare

Sistemul PhotoMatch servește un singur tip de actor uman — utilizatorul înregistrat — și interacționează cu două sisteme externe: serverul de email SMTP pentru livrarea pozelor și API-ul Google Gemini pentru detecția alternativă de conținut sensibil. Accesul la toate funcționalitățile aplicative este condiționat de autentificare, cu excepția ecranelor de înregistrare și login.

Cazurile de utilizare principale se grupează în șase domenii funcționale. Domeniul de autentificare cuprinde înregistrarea unui cont nou cu email și parolă și autentificarea cu obținerea token-ului JWT. Domeniul de procesare imagine include procesarea individuală (capturare sau selectare foto, vectorizare locală, corecție CLAHE și LUT), procesarea în lot a unui folder întreg (până la 100 de imagini), procesarea burst cu selecție automată a celei mai bune imagini din grupuri similare, și căutarea ghidată de stil cu upload de imagini de referință. Domeniul de anonimizare cuprinde detecția regiunilor sensibile și vizualizarea bounding boxes, aplicarea blurul-ului Gaussian și exportul imaginii anonimizate. Domeniul de analiză facială include detectarea și gruparea fețelor din poze, iar cel de livrare automatizată presupune distribuția pozelor de la un eveniment către participanți pe baza recunoașterii faciale. Scraping-ul web cu evaluare de risc și gestionarea favoritelor completează tabloul funcțional.

---

### Descrierea diagramei — Diagrama de cazuri de utilizare

**Scopul diagramei:** să reprezinte complet și sintetic toate funcționalitățile expuse utilizatorului, fără a intra în detalii de implementare. Este prima diagramă pe care membrii comisiei o vor consulta pentru a înțelege ce face aplicația.

**De ce este necesară:** standardul IEEE pentru documentarea software impune un use case diagram ca punct de plecare al proiectării; în contextul lucrării de licență, demonstrează că studentul a gândit în termeni de cerințe funcționale înainte de a implementa.

**Tipul diagramei:** Use Case Diagram UML 2.x.

**Nivelul de granularitate recomandat:** mediu-fin — fiecare funcționalitate vizibilă utilizatorului apare ca un caz de utilizare separat; operațiunile interne ale serverului nu apar.

**Actori:**
- `Utilizator` (actor primar, elipsă în stânga diagramei) — persoana care folosește aplicația Android
- `Server SMTP` (actor secundar, elipsă în dreapta) — sistem extern pentru livrarea email-urilor
- `Google Gemini API` (actor secundar, elipsă în dreapta) — sistem extern pentru detecția alternativă

**Cazuri de utilizare și relații:**

*Grup Autentificare (dreptunghi de sistem):*
- `Înregistrare cont` — include `Validare email și parolă`
- `Autentificare` — returnează token JWT; toate celelalte UC au relație «extend» condiționată de autentificare

*Grup Corecție imagine:*
- `Procesare imagine individuală` — include `Vectorizare CLIP locală`, include `Detecție defecte locală`, include `Căutare FAISS`, include `Aplicare CLAHE`, include `Aplicare 3D LUT`
- `Procesare lot` — extinde `Procesare imagine individuală` cu multiplicitate (max 100 imagini)
- `Procesare burst` — include `Clustering local`, include `Selecție automată`
- `Căutare ghidată de stil` — include `Upload imagini referință stil`
- `Salvare în favorite` — asociere cu `Procesare imagine individuală`
- `Vizualizare favorite` — cazul de utilizare independent
- `Comparare imagini` — asociere cu rezultatele procesării

*Grup Anonimizare:*
- `Anonimizare imagine` — include `Detecție regiuni sensibile (local)` SAU include `Detecție regiuni sensibile (Gemini)` (relație «extend» cu punct de extensie `alegere detector`)
- `Detecție regiuni sensibile (local)` — asociere cu `Google Gemini API` absentă; include `Detectie ecrane YOLOv8`, include `Detecție carduri badge_model`
- `Detecție regiuni sensibile (Gemini)` — asociere cu `Google Gemini API`
- `Export imagine anonimizată` — extinde `Anonimizare imagine`

*Grup Analiză facială și livrare:*
- `Grupare fețe` — include `Detecție față MLKit`, include `Embedding FaceNet`, include `Clustering greedy`
- `Livrare automată poze` — include `Scraping pagină angajați`, include `Grupare fețe`, include `Matching centroid-angajat`, include `Trimitere email`; asociere cu `Server SMTP`
- `Configurare versiune delivery (v1/v2)` — extinde `Livrare automată poze`

*Grup Web Scraper:*
- `Scraping web cu detecție riscuri` — include `Extragere imagini pagină web`, include `Calcul risk score`, include `Filtrare după nivel risc`
- `Blur inline imagine scrapată` — extinde `Scraping web cu detecție riscuri`
- `Deschidere în Blur Editor` — extinde `Scraping web cu detecție riscuri`

**Ce trebuie evidențiat vizual:** granița sistemului (dreptunghi care încorporează toate UC), separarea pe grupuri tematice prin note sau culori, relațiile «include» față de «extend».

**Informații care nu trebuie omise:** relația cu cei doi actori externi (SMTP și Gemini), condiția de autentificare, faptul că detecția locală și cea Gemini sunt alternative (nu cumulative).

**Specificație pentru realizarea diagramei:**
Desenați un dreptunghi mare etichetat „PhotoMatch System". În stânga sa plasați actorul „Utilizator" (om stilizat). În dreapta plasați „Server SMTP" și „Google Gemini API". În interiorul dreptunghiului, organizați elipsele pe șase coloane/grupuri vizuale: Autentificare, Corecție, Anonimizare, Analiză facială, Livrare, Scraper. Conectați „Utilizator" la toate UC-urile cu linii de asociere. Desenați relații «include» (linie punctată cu săgeată, etichetă «include») de la UC-urile compuse spre UC-urile incluse. Desenați relații «extend» (linie punctată cu săgeată, etichetă «extend») de la UC-urile opționale/condiționate spre UC-ul extins. Conectați „Anonimizare imagine" la „Google Gemini API" prin „Detecție Gemini". Conectați „Livrare automată poze" la „Server SMTP". UC-urile de autentificare nu se conectează la actorii externi.

---

## 3.4 Diagrama de clase

Arhitectura aplicației Android se reflectă în trei straturi principale de clase: stratul de prezentare (Activities), stratul de comunicare (ApiClient, ApiService, modele request/response) și stratul de persistență (Room entities și DAO).

Ierarhia de bază `BaseActivity → BaseLocalActivity → BaseServerActivity` a fost descrisă în secțiunea 3.2.1. Activitățile concrete, cum ar fi `ProcessingActivity`, `BlurActivity`, `WebScraperActivity`, `DeliveryActivity` sau `FaceGroupsActivity`, extind `BaseServerActivity` și folosesc câmpurile și metodele moștenite fără a le suprascrie, respectând principiul deschis-închis.

Clasa `ApiClient` este implementată ca singleton cu instanță statică privată și metodă `getInstance()` cu sincronizare. Ea expune metoda statică `service()` care returnează instanța `ApiService` și metodele utilitare `saveServerIp()`, `clearToken()`, `isLoggedIn()`. Interfața `ApiService`, adnotată cu Retrofit, definește toate metodele de comunicare grupate pe domenii funcționale.

Modelele de date request/response sunt clase POJO cu câmpuri publice, deserializate de Gson. Exemple reprezentative: `SearchAndCorrectRequest` (câmpuri `vector: List<Float>`, `topK: int`), `BlurDetectResponse` (câmpuri `regions: List<BlurRegion>`, `imageWidth: int`, `imageHeight: int`), `ScrapeResponse` (câmpuri `results: List<ScrapedImageResult>`, `imagesWithRisk: int`, `imagesProcessed: int`), `DeliveryResponse` (câmpuri `clustersFound: int`, `matched: int`, `sent: int`, `failed: int`).

Pe server, ierarhia de modele Pydantic oglindește structura clientului: `BlurRegionItem`, `BlurDetectResponse`, `ScrapeRequest`, `ScrapedImageResult`, `ScrapeResponse` sunt clase Pydantic care validează și serializează automat datele.

---

### Descrierea diagramei — Diagrama de clase

**Scopul diagramei:** să prezinte structura statică a codului Android, clasele principale, relațiile dintre ele și distribuția responsabilităților. Este instrumentul prin care comisia evaluează calitatea proiectării orientate-obiect.

**De ce este necesară:** demonstrează că aplicația a fost proiectată cu principii OOP clare — ierarhii de moștenire, separare a responsabilităților, pattern-uri de design.

**Tipul diagramei:** Class Diagram UML 2.x.

**Nivelul de granularitate recomandat:** mediu — atributele și metodele publice și protejate relevante sunt vizibile; metodele private de implementare internă pot fi omise dacă aglomerează diagrama.

**Clase și membri principali:**

`AppCompatActivity` (din librăria AndroidX, reprezentată ca clasă externă):
- Superclasă pentru toată ierarhia

`BaseActivity` (abstractă):
- Atribute: niciun atribut de instanță propriu
- Metode: `#requiresAuth(): boolean`, `+api(): ApiService`, `+showError(msg: String)`, `+showServerIpDialog()`, `+logout()`
- Relație: extinde `AppCompatActivity`

`BaseLocalActivity` (abstractă):
- Atribute: `#executor: ExecutorService`, `#primaryButton: Button`, `#progressBar: ProgressBar`
- Metode: `+setProcessing(processing: boolean)`
- Relație: extinde `BaseActivity`

`BaseServerActivity` (abstractă):
- Fără atribute sau metode proprii (marker semantic)
- Relație: extinde `BaseLocalActivity`

Activities concrete (fiecare extinde `BaseServerActivity`):
- `ProcessingActivity`: atribute `clipEncoder: CLIPEncoder`, `defectDetector: DefectDetector`; metoda `processImage(uri: Uri)`
- `BlurActivity`: atribute `origBoxed: Map<Integer,Bitmap>`, `blurred: Map<Integer,Bitmap>`, `showBlurred: boolean`; metode `processCurrentImage()`, `drawBoxes(src: Bitmap, regions: List<BlurRegion>, scaleX: float, scaleY: float): Bitmap`, `toggleView()`, `exportCurrentBlurred()`
- `WebScraperActivity`: atribute `allResults: List<ScrapedImageResult>`, `displayedResults: List<ScrapedImageResult>`, `activeFilter: String`; metode `startScrape()`, `applyFilter()`, `applyBlurInline(item: ScrapedImageResult)`, `openInBlurEditor(item: ScrapedImageResult)`
- `DeliveryActivity`: atribut `useV2: boolean`; metoda `runDelivery()`
- `FaceGroupsActivity`: metoda `clusterFaces()`
- `BatchActivity`, `BurstActivity`, `ClusterActivity`, `StyleSetupActivity`, `ResultsActivity`, `FavoritesActivity`, `CompareActivity`, `LoginActivity`, `RegisterActivity` — menționate cu rol și superclasă, fără detalii complete pentru lizibilitate

`ApiClient` (singleton):
- Atribute: `-instance: ApiClient` (static), `-apiService: ApiService`
- Metode: `+getInstance(): ApiClient` (static), `+service(): ApiService` (static), `+init(context: Context)` (static), `+saveServerIp(ip: String)`, `+getServerIp(): String`, `+isLoggedIn(): boolean`, `+clearToken()`
- Relație: agregare cu `ApiService` (1 la 1)

`ApiService` (interfață Retrofit):
- Metode reprezentative: `blurDetect(file: Part, detector: String): Call<BlurDetectResponse>`, `blurSensitive(file: Part, detector: String): Call<ResponseBody>`, `scrapeImages(body: ScrapeRequest): Call<ScrapeResponse>`, `searchAndCorrect(req: SearchAndCorrectRequest, weight: float, includeImages: boolean): Call<SearchAndCorrectResponse>`, `deliveryRun(url: RequestBody, photos: List<Part>): Call<DeliveryResponse>`

Modele POJO:
- `BlurRegion`: `label: String`, `x: int`, `y: int`, `w: int`, `h: int`
- `BlurDetectResponse`: `regions: List<BlurRegion>`, `imageWidth: int`, `imageHeight: int`
- `ScrapedImageResult`: `sourceUrl: String`, `thumbnailB64: String`, `originalWidth: int`, `originalHeight: int`, `thumbnailWidth: int`, `thumbnailHeight: int`, `regions: List<BlurRegion>`, `riskScore: float`, `riskLabel: String`
- `ScrapeRequest`: `url: String`, `detector: String`
- `ScrapeResponse`: `results: List<ScrapedImageResult>`, `imagesWithRisk: int`, `imagesProcessed: int`, `skippedCount: int`
- `SearchAndCorrectRequest`: `vector: List<Float>`, `topK: int`
- `DeliveryResponse`: `clustersFound: int`, `matched: int`, `sent: int`, `failed: int`

Entități Room:
- `FavoritePhoto` (entitate Room): `id: int` (PK, autogenerat), `originalBase64: String`, `correctedBase64: String`, `uriString: String`, `retrieved: String`, `timestamp: long`, `improvements: String` (JSON)
- `FavoriteDao` (interfață DAO): `insert(f: FavoritePhoto)`, `delete(f: FavoritePhoto)`, `getAll(): List<FavoritePhoto>`, `findByRetrieved(name: String): FavoritePhoto`
- `AppDatabase` (abstractă, RoomDatabase): `favoriteDao(): FavoriteDao`; singleton prin `get(ctx: Context): AppDatabase`

**Relații:**
- `BaseActivity` —extends→ `AppCompatActivity`
- `BaseLocalActivity` —extends→ `BaseActivity`
- `BaseServerActivity` —extends→ `BaseLocalActivity`
- Toate activitățile concrete —extends→ `BaseServerActivity`
- `BaseActivity` —uses (dependency)→ `ApiClient`
- `ApiClient` —aggregates→ `ApiService` (1 la 1)
- `BlurActivity` —uses→ `BlurDetectResponse`, `BlurRegion`
- `WebScraperActivity` —uses→ `ScrapeRequest`, `ScrapeResponse`, `ScrapedImageResult`
- `ResultsActivity` —uses→ `AppDatabase`, `FavoriteDao`, `FavoritePhoto` (associere)
- `FavoritesActivity` —uses→ `AppDatabase`, `FavoriteDao`, `FavoritePhoto`
- `AppDatabase` —aggregates→ `FavoriteDao`
- `FavoriteDao` —manages→ `FavoritePhoto`

**Ce trebuie evidențiat vizual:** ierarhia de moștenire cu stereotipul `<<abstract>>` pe cele trei clase de bază, pattern-ul Singleton pe `ApiClient`, separarea clară între stratul de prezentare și cel de date.

**Specificație pentru realizarea diagramei:**
Organizați clasele pe patru zone orizontale. Sus: ierarhia de bază (`AppCompatActivity` → `BaseActivity` → `BaseLocalActivity` → `BaseServerActivity`), conectate prin săgeți de moștenire (triunghi gol). Sub ierarhia de bază: activitățile concrete — plasați-le în arc, conectate prin săgeți de moștenire la `BaseServerActivity`. La dreapta: `ApiClient` și `ApiService` conectate prin linie de agregare cu multiplicitate 1—1. Sub activitățile concrete: modelele POJO, grupate în două subgrupuri — „Blur models" și „Scrape models". Jos: entitățile Room — `AppDatabase`, `FavoriteDao`, `FavoritePhoto` conectate prin compoziție (romb plin). Marcați `BaseActivity`, `BaseLocalActivity`, `BaseServerActivity` cu stereotipul `<<abstract>>`. Marcați `ApiService` cu stereotipul `<<interface>>`. Marcați `FavoriteDao` cu stereotipul `<<interface>>`. Marcați `ApiClient` cu nota „<<Singleton>>".

---

## 3.5 Structura bazei de date

Sistemul folosește trei mecanisme distincte de persistență, fiecare ales în funcție de natura datelor gestionate.

**Baza de date SQLite a aplicației Android (Room)** conține un singur tabel, `favorites`, gestionat prin abstractizarea Room. Tabelul stochează pozele marcate ca favorite de utilizator după procesare. Coloana `id` este cheie primară de tip INTEGER cu auto-incrementare. Coloana `original_base64` de tip TEXT conține imaginea originală codificată Base64. Coloana `corrected_base64` stochează versiunea corectată. Coloana `uri_string` păstrează URI-ul sursei pentru referință. Coloana `retrieved` conține numele bazename al imaginii de referință recuperate din setul FiveK, folosit ca identificator unic pentru a evita duplicatele (interogat prin `findByRetrieved()`). Coloana `timestamp` de tip INTEGER stochează momentul salvării în milisecunde Unix, folosit pentru sortare descrescătoare. Coloana `improvements` de tip TEXT stochează un obiect JSON cu scorurile de îmbunătățire per categorie de defect.

**Baza de date SQLite a serverului (SQLAlchemy)** conține un singur tabel, `users`. Coloana `id` este cheie primară INTEGER cu auto-incrementare. Coloana `email` de tip VARCHAR este unică și indexată, reprezentând identificatorul de autentificare. Coloana `hashed_password` de tip VARCHAR stochează hash-ul bcrypt al parolei — parola în clar nu este niciodată persistată. Coloana `created_at` de tip DATETIME este populată automat la inserție. Coloana `is_active` de tip BOOLEAN (implicit `True`) permite dezactivarea administrativă a conturilor fără ștergere.

**Indexul FAISS** nu este o bază de date relațională în sens clasic, ci un index vectorial rezident în memorie de tip `IndexFlatIP` (produs intern pe vectori L2-normalizați, echivalent cu similaritate cosinus). Indexul conține 3.499 de vectori de dimensiune 517, corespunzând celor 3.499 de imagini din setul MIT FiveK. Fiecare vector hibrid concatenează embedding-ul CLIP ViT-B/32 ponderat cu factorul 1.0 (512 dimensiuni) și vectorul de scoruri de defecte ponderat cu factorul 5.0 (5 dimensiuni), re-normalizat L2. Maparea între indexul FAISS și numele fișierelor de pe disc este stocată în aceeași arhivă `hybrid_vectors.npz`.

---

### Descrierea diagramei — Diagrama entitate-relație (structura bazei de date)

**Scopul diagramei:** să prezinte formal structura tabelelor, atributele, tipurile de date, cheile și relațiile dintre entități.

**De ce este necesară:** demonstrează proiectarea corectă a nivelului de persistență și justifică deciziile de normalizare sau denormalizare.

**Tipul diagramei:** Entity-Relationship Diagram (ERD) sau diagramă de tabele relaționale.

**Nivelul de granularitate:** fin — toate coloanele, tipurile de date, constrângerile și cheile trebuie vizibile.

**Entități și atribute:**

Entitatea `favorites` (baza Android):
- `id: INTEGER, PK, AUTOINCREMENT, NOT NULL`
- `original_base64: TEXT, NOT NULL`
- `corrected_base64: TEXT, NOT NULL`
- `uri_string: TEXT, NOT NULL`
- `retrieved: TEXT, NOT NULL, UNIQUE`
- `timestamp: INTEGER, NOT NULL`
- `improvements: TEXT` (JSON, nullable)

Entitatea `users` (baza server):
- `id: INTEGER, PK, AUTOINCREMENT, NOT NULL`
- `email: VARCHAR(255), NOT NULL, UNIQUE, INDEX`
- `hashed_password: VARCHAR(255), NOT NULL`
- `created_at: DATETIME, NOT NULL, DEFAULT CURRENT_TIMESTAMP`
- `is_active: BOOLEAN, NOT NULL, DEFAULT TRUE`

**Relații:** cele două baze de date sunt independente și nu au relații între ele. Nu există chei externe cross-database. Ambele sunt baze SQLite separate fizic.

**Ce trebuie evidențiat vizual:** tipul de date al fiecărei coloane, marcarea PK (cheie primară), marcarea coloanelor UNIQUE și NOT NULL, absența relațiilor între cele două baze.

**Specificație pentru realizarea diagramei:**
Desenați două dreptunghiuri independente, fiecare reprezentând un tabel. Etichetați primul „favorites (Android — Room)" și al doilea „users (Server — SQLAlchemy)". În fiecare dreptunghi, listați coloanele cu formatul: `<cheie_icon> <nume_coloana> : <TIP_DATE> [constrângeri]`. Marcați coloanele PK cu o cheie aurie. Marcați coloanele UNIQUE cu `(U)`. Marcați coloanele NOT NULL cu `NN`. Adăugați o casetă informativă separată pentru FAISS Index: „FAISS IndexFlatIP (in-memory)" cu proprietățile: dimensiune vector = 517, număr vectori = 3.499, metrica = Inner Product (echivalent cosine pe vectori L2-normalizați), persistare = hybrid_vectors.npz. Conectați FAISS la o notă care explică maparea index → basename fișier FiveK. Nu desenați relații între cele două tabele SQLite — subliniați explicit că sunt baze de date separate.

---

## 3.6 Fluxul de funcționare al aplicației

### 3.6.1 Fluxul corecției de culoare (funcționalitatea principală)

Utilizatorul selectează o fotografie din galerie sau captează una cu camera. Aplicația, în `ProcessingActivity`, declanșează două inferențe paralele pe modelele TFLite locale: CLIP ViT-B/32 (care produce un embedding semantic de 512 de dimensiuni) și capul de detecție defecte bazat pe MobileNetV3Small (care produce 5 scoruri sigmoid pentru blur, noise, overexposure, underexposure, compression). Clasa `HybridVectorBuilder` concatenează cele două rezultate după aplicarea ponderilor CLIP_WEIGHT=1.0 și DEFECT_WEIGHT=5.0, normalizează L2 vectorul rezultant de 517 dimensiuni și îl trimite la server prin `POST /search_and_correct` ca body JSON.

Pe server, vectorul este folosit pentru o căutare `IndexFlatIP` în FAISS care returnează top-5 imagini similare. Cel mai bun match este selectat după un scor combinat de similaritate vectorială și scor estetic (calculat pe baza netezimii, expunerii și contrastului imaginii de referință). Serverul recuperează perechea RAW și editat a imaginii câștigătoare, extrage un LUT 3D pe o grilă de 17³ prin interpolare LinearNDInterpolator pe 50.000 de pixeli eșantionați aleatoriu, și aplică CLAHE pe canalul L al imaginii de intrare în spațiul LAB. Corecția finală combină imaginea CLAHE cu LUT-ul aplicat printr-o amestecare moderată (factor 0.2), returnând trei imagini Base64: originalul, versiunea CLAHE și versiunea finală.

### 3.6.2 Fluxul anonimizării

Utilizatorul selectează una sau mai multe imagini în `BlurActivity` și apasă butonul BLUR. Aplicația citește bytes-urile brute ale imaginii (fără resampling) și face două request-uri secvențiale. Primul, `POST /blur-detect`, returnează lista de regiuni detectate cu coordonate în spațiul imaginii originale și dimensiunile originalului. Android calculează factorii de scalare `scaleX = bitmap.width / serverWidth` și `scaleY = bitmap.height / serverHeight`, desenează bounding boxes amber pe un bitmap downsampled la 1200px și afișează imaginea cu boxes. Al doilea request, `POST /blur-sensitive`, repetă detecția și returnează direct imaginea JPEG cu blur aplicat. Utilizatorul poate toggle între cele două vederi.

### 3.6.3 Fluxul livrării automate

Utilizatorul furnizează URL-ul unei pagini HTML cu carduri de angajați și selectează un folder cu pozele de la eveniment. Serverul scrapează pagina, extrage numele, email-urile și fotografiile angajaților și construiește o bază de embeddings faciale cu FaceNet TFLite. Pe pozele de la eveniment, serverul (versiunea v2) rulează detecție MLKit, embeddinguri FaceNet și un algoritm de clustering greedy cu prag de similaritate cosinus 0.75, obținând grupe de fețe. Centroidul fiecărui cluster (mediana embeddings-urilor, re-normalizat L2) este comparat cu fiecare angajat din baza de date prin similaritate cosinus, cu prag 0.70. Fotografiile din clusterele matched sunt trimise prin SMTP ca atașamente la adresele email corespunzătoare.

---

### Descrierea diagramei — Diagrama de activitate pentru fluxul principal (corecție culoare)

**Scopul diagramei:** să descrie cronologic și ramificat pașii parcurși de o cerere de corecție de la acțiunea utilizatorului până la afișarea rezultatului, incluzând condițiile de eroare.

**De ce este necesară:** diagrama de activitate este cel mai potrivit instrument UML pentru a descrie un algoritm cu ramificații și paralelisme, esențial pentru a justifica deciziile de proiectare ale pipeline-ului hibrid local+server.

**Tipul diagramei:** Activity Diagram UML 2.x cu swimlanes.

**Nivelul de granularitate:** mediu-fin — fiecare pas logic distinct apare ca acțiune separată; detaliile de implementare (bucle for, alocare memorie) nu apar.

**Swimlanes:**
- `Utilizator`: acțiunile UI
- `Android (local)`: procesare pe dispozitiv
- `Server FastAPI`: procesare pe server

**Fluxul de activitate:**

1. (Utilizator) Selectare imagine din galerie sau captare camera → (Android) Citire bytes imagine
2. (Android) Fork paralel:
   - Ramura A: Inferență CLIP TFLite → embedding 512-dim
   - Ramura B: Inferență Defect Head TFLite → scoruri 5-dim
3. (Android) Join (așteptare ambele ramuri) → Construire vector hibrid 517-dim (ponderare + normalizare L2)
4. (Android) POST /search_and_correct cu vectorul JSON → (Server) Recepție request
5. (Server) Căutare FAISS top-5 → Selecție best match (scor similarity + estetic)
6. (Server) Recuperare pereche RAW/editat → Extragere LUT 3D (17³ grid, LinearNDInterpolator)
7. (Server) Aplicare CLAHE pe input (canal L, LAB) → Aplicare LUT → Amestec moderat (factor 0.2)
8. (Server) Codificare Base64 (original, CLAHE, final) → Răspuns JSON
9. (Android) Decodificare Bitmap din Base64 → Afișare în UI cu tab-uri (original / CLAHE / final)
10. (Utilizator) Opțional: Salvare în Favorite → (Android) Inserare Room DB

**Condiții de eroare (Decision nodes):**
- La pasul 4: dacă serverul nu răspunde → afișare Toast eroare
- La pasul 5: dacă FAISS nu e inițializat (fișier lipsă) → HTTP 503 → eroare UI
- La pasul 9: dacă Base64 invalid → afișare imagine placeholder

**Ce trebuie evidențiat vizual:** fork/join pentru paralelism local, swimlane-ul clar între Android și Server, nodurile de decizie pentru erori.

**Specificație pentru realizarea diagramei:**
Desenați trei swimlane-uri verticale: „Utilizator" (stânga), „Android (local)" (centru), „Server FastAPI" (dreapta). Începeți cu un nod de start (cerc plin) în swimlane „Utilizator". Conectați „Selectare imagine" → „Citire bytes" (Android). Desenați un nod Fork (bara orizontală neagră groasă) care separă două fluxuri paralele în swimlane Android: „Inferență CLIP TFLite" și „Inferență Defect Head TFLite". Reuniți-le într-un nod Join (bară similară). Continuați cu „Construire vector hibrid". Traversați granița spre Server cu „POST /search_and_correct". În Server: „Căutare FAISS" → „Selecție best match" → fork pentru „Recuperare RAW+editat" și „Calcul scor estetic" → join → „Extragere LUT 3D" → „CLAHE + LUT + amestec" → „Codificare Base64". Răspunsul revine în Android: „Decodificare Bitmap" → „Afișare UI". Adăugați un diamond de decizie după „POST" pentru cazul de eroare de rețea. Terminați cu nod de final (cerc plin cu contur).

---

*Notă: Toate diagramele descrise în acest capitol vor fi realizate în Enterprise Architect și incluse ca figuri numerotate în versiunea finală a lucrării.*
