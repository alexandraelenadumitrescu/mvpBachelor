# PhotoMatch V2 — Ramura `refactorizare`

> Această ramură extinde versiunea de bază cu autentificare JWT, integrarea BlurAPI, trimitere email din Android și securizarea tuturor endpoint-urilor. Conținutul de mai jos reprezintă **Capitolul 3 — Proiectarea arhitecturii soluției** documentat conform stării actuale a proiectului.

---

## Cuprins

1. [Ce aduce ramura `refactorizare`](#1-ce-aduce-ramura-refactorizare)
2. [Stack tehnologic](#2-stack-tehnologic)
3. [Capitolul 3 — Proiectarea arhitecturii soluției](#3-capitolul-3--proiectarea-arhitecturii-soluției)
   - [3.1 Arhitectura generală a soluției](#31-arhitectura-generală-a-soluției)
   - [3.2 Diagrama de cazuri de utilizare](#32-diagrama-de-cazuri-de-utilizare)
   - [3.3 Diagrama de clase](#33-diagrama-de-clase)
   - [3.4 Structura bazei de date](#34-structura-bazei-de-date)
4. [Instalare și rulare](#4-instalare-și-rulare)
5. [API Reference](#5-api-reference)
6. [Structura proiectului](#6-structura-proiectului)

---

## 1. Ce aduce ramura `refactorizare`

| Componentă | Adăugat față de `main` |
|---|---|
| `server/auth/` | Modul JWT complet: register / login / me, bcrypt, SQLite users.db |
| `server/blur_api/` | BlurAPI integrat în server: Gemini Vision + YOLOv8 local + benchmark |
| `server_v2.py` | Toate endpoint-urile protejate cu JWT; endpoint nou `/mail/send` |
| `android/api/` | `ApiClient` cu `AuthInterceptor`; `ApiService` cu login/register/mail; 4 modele noi |
| Refactorizare server | Helper-e `_load_image()`, `_correct_image()`, `DEFECT_NAMES` — −18 linii cod duplicat |

---

## 2. Stack tehnologic

**Server (Python 3.11)**

| Bibliotecă | Rol |
|---|---|
| FastAPI + Uvicorn | Framework HTTP și server ASGI |
| PyTorch 2.3 + TorchVision | CLIP ViT-B-32 + MobileNetV3Small defect head |
| OpenCLIP 2.24 | Encodare imagini CLIP |
| FAISS-CPU 1.8 | Index de similaritate vectorială |
| OpenCV 4.9 + Pillow | Procesare imagini, CLAHE, blur |
| SciPy + scikit-learn | Interpolare LUT 3D, K-Means clustering |
| google-generativeai | Gemini Vision API (BlurAPI) |
| ultralytics | YOLOv8 nano (detector local BlurAPI) |
| python-jose + passlib | JWT HS256 + bcrypt hashing |
| SQLAlchemy | ORM SQLite pentru users.db |
| smtplib | Trimitere email SMTP cu atașamente |

**Android (Java, min SDK 26)**

| Bibliotecă | Rol |
|---|---|
| Retrofit 2.9 + OkHttp | Client HTTP, `AuthInterceptor` JWT |
| TensorFlow Lite 2.13 | CLIP + defect head on-device |
| Google MLKit | Detecție facială on-device |
| Room 2.6 | Baza de date locală favorite |
| Glide 4.16 | Încărcare și cache imagini |
| Material Design 3 | UI components |

---

## 3. Capitolul 3 — Proiectarea arhitecturii soluției

Soluția este compusă din **trei componente principale** care se completează reciproc:

| # | Componentă | Scop |
|---|---|---|
| 1 | **PhotoMatch** | Corecție automată fotografii — Android + server FastAPI |
| 2 | **BlurAPI** | Detecție și blurare conținut sensibil — Gemini Vision + YOLOv8 |
| 3 | **PhotoMailer** | Distribuție automată poze eveniment — FaceNet + scraping + email |

---

### 3.1 Arhitectura generală a soluției

#### Diagrama de ansamblu

```mermaid
flowchart TD
    subgraph ANDROID["📱 Android App (Java)"]
        direction LR
        ML["On-Device ML\nCLIP ViT-B-32\nMobileNetV3Small"]
        AC["ApiClient\nOkHttp + AuthInterceptor"]
        UI["Activities\nPhotoMatch / Blur / Delivery"]
        DB_LOCAL["Room SQLite\nFavorite Photos"]
        ML -->|"517-dim vector"| AC
        UI --> AC
        UI --> DB_LOCAL
    end

    subgraph SERVER["🖥️ FastAPI Server (Python)"]
        direction TB
        AUTH_MOD["auth/\nJWT Register/Login\nSQLite users.db"]
        PM_MOD["PhotoMatch\nFAISS + CLAHE + LUT\n3.499 vectori referință"]
        BLUR_MOD["BlurAPI\nGemini Vision\nYOLOv8 local"]
        MAIL_MOD["Mail\nSMTP STARTTLS\nMIME attachments"]
    end

    subgraph PHOTOMAILER["🐍 PhotoMailer CLI (Python)"]
        direction LR
        SCRAPE["Scraper\nBeautifulSoup"]
        FACEDB["Face DB\nFaceNet 512-dim\npickle"]
        MATCHER["Matcher\nCosine sim ≥ 0.60"]
        MAILER_CLI["Mailer\nsmtplib bulk"]
        SCRAPE --> FACEDB --> MATCHER --> MAILER_CLI
    end

    subgraph EXTERNAL["☁️ Servicii externe"]
        GEMINI["Gemini Vision API\ngemini-1.5-flash"]
        SMTP_SRV["SMTP Server\nGmail / orice provider"]
    end

    AC -->|"HTTP + Bearer JWT"| SERVER
    AUTH_MOD -->|"Depends(get_current_active_user)\nprotejează toate endpoint-urile"| PM_MOD
    AUTH_MOD --> BLUR_MOD
    AUTH_MOD --> MAIL_MOD
    BLUR_MOD -->|"imagine → bounding boxes"| GEMINI
    MAIL_MOD -->|"email + atașamente"| SMTP_SRV
    MAILER_CLI --> SMTP_SRV
```

#### Fluxul principal PhotoMatch (sequence diagram)

```mermaid
sequenceDiagram
    actor U as Utilizator
    participant A as Android App
    participant S as FastAPI Server
    participant F as FAISS Index

    U->>A: Selectează fotografie
    A->>A: CLIP encode → 512-dim
    A->>A: MobileNetV3 → 5-dim defect scores
    A->>A: Concat + normalize → 517-dim vector
    A->>S: POST /auth/login (la prima utilizare)
    S-->>A: JWT token
    A->>S: POST /search_and_correct {vector: [...517 float-uri]}\n+ Authorization: Bearer <token>
    S->>F: index.search(query, top_k=10)
    F-->>S: [(basename, similarity), ...]
    S->>S: _aesthetic_rerank() → cel mai bun match
    S-->>A: {retrieved, similarity, lut_cached, raw_b64, edited_b64}
    A->>S: GET /lut/{basename} (dacă nu e în cache local)
    S-->>A: LUT 17³ × 3 float32 bytes (base64)
    A->>A: CLAHE + aplicare LUT local
    A->>U: Afișează imaginea corectată
```

#### Principiul privacy-first

```mermaid
flowchart LR
    IMG["📷 Imaginea utilizatorului"]
    CLIP_ONDEV["CLIP on-device\n(TFLite)"]
    MNV3_ONDEV["MobileNetV3\non-device (TFLite)"]
    VEC["517-dim\nvector numeric"]
    SERVER["Server\n(primește DOAR numere)"]
    LUT_DL["LUT descărcat o dată\ncached local"]
    CORRECT["Corecție aplicată\nlocal pe telefon"]

    IMG --> CLIP_ONDEV
    IMG --> MNV3_ONDEV
    CLIP_ONDEV --> VEC
    MNV3_ONDEV --> VEC
    VEC -->|"numai vectorul pleacă din telefon"| SERVER
    SERVER --> LUT_DL
    LUT_DL --> CORRECT
    IMG -->|"rămâne pe telefon"| CORRECT

    style IMG fill:#c8e6c9
    style VEC fill:#fff9c4
    style SERVER fill:#bbdefb
    style CORRECT fill:#c8e6c9
```

> **Excepție:** endpoint-urile `/blur-sensitive` și `/mail/send` trimit imaginea la server — utilizatorul este informat explicit.

---

### 3.2 Diagrama de cazuri de utilizare

```mermaid
flowchart TD
    ACT1["👤 Utilizator autentificat"]
    ACT2["🔧 Administrator CLI"]
    ACT_GEM["☁️ Gemini Vision API"]
    ACT_SMTP["📧 SMTP Server"]

    subgraph UC_AUTH["Autentificare"]
        UC1("Înregistrare cont")
        UC2("Autentificare / Login")
        UC3("Vizualizare profil")
    end

    subgraph UC_PM["PhotoMatch"]
        UC4("Corecție fotografie unică")
        UC5("Procesare batch\npână la 100 imagini")
        UC6("Mod burst\nclusterizare + best-shot")
        UC7("Căutare cu stil personalizat")
        UC8("Clustering fotografii")
        UC9("Grupare după persoane\n(detecție facială)")
        UC10("Gestionare favorite")
    end

    subgraph UC_BLUR["BlurAPI"]
        UC11("Blurare conținut sensibil\ndetector Gemini")
        UC12("Blurare conținut sensibil\ndetector local YOLOv8")
    end

    subgraph UC_MAIL["Email"]
        UC13("Trimitere fotografii\ndin Android")
    end

    subgraph UC_CLI["PhotoMailer CLI"]
        UC14("Scraping angajați\nde pe pagina web")
        UC15("Construire bază\nde date faciale")
        UC16("Matching fețe\nîn poze eveniment")
        UC17("Distribuție automată\nemail bulk")
    end

    ACT1 --> UC_AUTH
    ACT1 --> UC_PM
    ACT1 --> UC_BLUR
    ACT1 --> UC_MAIL
    ACT2 --> UC_CLI

    UC11 -.->|"include"| ACT_GEM
    UC13 -.->|"include"| ACT_SMTP
    UC17 -.->|"include"| ACT_SMTP
```

---

### 3.3 Diagrama de clase

#### Diagrama 1 — Clientul Android (stratul API)

```mermaid
classDiagram
    class ApiClient {
        +String PREFS_NAME
        +String KEY_TOKEN
        +String DEFAULT_IP
        -Context appContext$
        -ApiClient instance$
        -ApiService apiService
        +init(Context)$
        +getInstance() ApiClient$
        +saveToken(String)$
        +getToken() String$
        +clearToken()$
        +isLoggedIn() bool$
        +getServerIp() String$
        +saveServerIp(String)$
        +getService() ApiService
        +base64ToBitmap(String) Bitmap$
        +bitmapToBase64(Bitmap) String$
    }

    class ApiService {
        <<interface>>
        +health() Call~Map~
        +register(UserCreate) Call~UserResponse~
        +login(email, password) Call~TokenResponse~
        +me() Call~UserResponse~
        +searchAndCorrect(req, weight, images) Call~SearchAndCorrectResponse~
        +styleSearch(StyleSearchRequest) Call~StyleSearchResponse~
        +styleVectors(StyleVectorsRequest) Call~StyleVectorsResponse~
        +getLut(basename) Call~LutResponse~
        +cluster(ClusterRequest) Call~ClusterResponse~
        +blurSensitive(MultipartBody.Part) Call~ResponseBody~
        +sendPhotos(toEmail, subject, message, files) Call~MailSendResponse~
    }

    class UserCreate {
        +String email
        +String password
    }

    class TokenResponse {
        +String access_token
        +String token_type
    }

    class UserResponse {
        +int id
        +String email
        +boolean is_active
        +String created_at
    }

    class MailSendResponse {
        +boolean sent
        +String to
        +int photos_attached
    }

    class SearchAndCorrectResponse {
        +String retrieved
        +float similarity
        +String raw_b64
        +String edited_b64
        +boolean lut_cached
        +float match_aesthetic_score
    }

    ApiClient --> ApiService : creates via Retrofit
    ApiService ..> UserCreate
    ApiService ..> TokenResponse
    ApiService ..> UserResponse
    ApiService ..> MailSendResponse
    ApiService ..> SearchAndCorrectResponse
```

#### Diagrama 2 — Modulul de autentificare server (`auth/`)

```mermaid
classDiagram
    class User {
        <<SQLAlchemy Model>>
        +int id
        +String email
        +String hashed_password
        +DateTime created_at
        +bool is_active
    }

    class Database {
        <<module: database.py>>
        +Engine engine
        +sessionmaker SessionLocal
        +DeclarativeBase Base
        +get_db() Generator~Session~
    }

    class AuthService {
        <<module: service.py>>
        +String JWT_SECRET
        +String ALGORITHM = "HS256"
        +int TOKEN_EXPIRE_MINUTES
        +hash_password(str) str
        +verify_password(str, str) bool
        +create_access_token(dict) str
        +authenticate_user(db, email, pass) User
    }

    class AuthDependencies {
        <<module: dependencies.py>>
        +OAuth2PasswordBearer oauth2_scheme
        +get_current_user(token, db) User
        +get_current_active_user(user) User
    }

    class AuthRouter {
        <<module: router.py>>
        +POST /auth/register(UserCreate, db) UserResponse
        +POST /auth/login(form, db) Token
        +GET /auth/me(current_user) UserResponse
    }

    class UserCreate {
        <<Pydantic Schema>>
        +EmailStr email
        +str password
    }

    class UserResponse {
        <<Pydantic Schema>>
        +int id
        +str email
        +bool is_active
        +datetime created_at
    }

    class Token {
        <<Pydantic Schema>>
        +str access_token
        +str token_type
    }

    Database --> User : manages
    AuthService --> User : queries/creates
    AuthDependencies --> AuthService : uses JWT_SECRET
    AuthDependencies --> User : returns
    AuthRouter --> AuthService : calls
    AuthRouter --> UserCreate : receives
    AuthRouter --> UserResponse : returns
    AuthRouter --> Token : returns
```

#### Diagrama 3 — PhotoMailer (pipeline CLI Python)

```mermaid
classDiagram
    class FaceDB {
        <<module: face_db.py>>
        +List MOCK_EMPLOYEES
        +build_db(employees) dict
        +save_db(db, path)
        +load_db(path) dict
    }

    class Scraper {
        <<module: scraper.py>>
        +mock_employees() List~dict~
        +scrape_employees(url) List~dict~
    }

    class Matcher {
        <<module: matcher.py>>
        +float THRESHOLD = 0.60
        +match_photos(photo_paths, face_db) dict
        -_best_match(embedding, face_db) tuple
        -_cosine_sim(a, b) float
    }

    class Mailer {
        <<module: mailer.py>>
        +send_photos(matches, smtp_cfg, dry_run)
        -_connect(host, port, user, pass) SMTP
        -_build_message(to, from, subject, paths) MIMEMultipart
    }

    class PhotoMailerCLI {
        <<module: main.py>>
        +--mock bool
        +--dry-run bool
        +--url str
        +--photos-dir str
        +--smtp-host str
        +--smtp-port int
        +--smtp-user str
        +--smtp-password str
    }

    PhotoMailerCLI --> Scraper : calls
    PhotoMailerCLI --> FaceDB : build/load
    PhotoMailerCLI --> Matcher : match_photos()
    PhotoMailerCLI --> Mailer : send_photos()
    Scraper --> FaceDB : provides employees
    FaceDB ..> Matcher : face embeddings dict
    Matcher ..> Mailer : matches dict
```

---

### 3.4 Structura bazei de date

#### Baza de date utilizatori — SQLite (`users.db`, SQLAlchemy)

Creată automat la pornirea serverului (`Base.metadata.create_all(bind=engine)`).

```mermaid
erDiagram
    USERS {
        INTEGER id PK "autoincrement"
        VARCHAR email UK "indexed, not null"
        VARCHAR hashed_password "bcrypt hash, not null"
        DATETIME created_at "default: utcnow()"
        BOOLEAN is_active "default: True"
    }
```

#### Baza de date locală Android — Room/SQLite

```mermaid
erDiagram
    FAVORITE_PHOTO {
        INTEGER id PK "autoincrement"
        TEXT imagePath "calea locală pe dispozitiv"
        TEXT retrievedBasename "basename din FiveK dataset"
        REAL similarity "scor FAISS 0.0–1.0"
        INTEGER timestamp "Unix millis"
    }
```

#### Vectorii de referință PhotoMatch — fișier NPZ (in-memory)

Nu este o bază de date relațională — datele sunt încărcate integral în RAM la startup și indexate în FAISS.

```
hybrid_vectors.npz
├── vectors  — shape (3499, 517)  float32  ~7 MB
└── names    — shape (3499,)      str      basename FiveK (ex. "a1234-Expert-C")

lut_cache    — dict {basename: np.ndarray (17,17,17,3)}  — completat în background
aesthetic_cache — dict {basename: float}                 — completat în background
style_profiles  — dict {session_id: np.ndarray (N,517)}  — per-request, în memorie
```

#### Baza de date faciale PhotoMailer — pickle

```
face_db.pickle
└── dict {
      "user@company.com": np.ndarray (512,)  ← embedding FaceNet
      "user2@company.com": np.ndarray (512,)
      ...
    }
```

> **Limitare de securitate:** fișierul pickle nu este criptat. Soluție recomandată pentru producție: criptare AES a fișierului înainte de stocare.

---

## 4. Instalare și rulare

### Server

```bash
# 1. Instalare dependențe
cd server
pip install -r requirements.txt

# 2. Variabile de mediu obligatorii
set JWT_SECRET_KEY=<un-secret-lung-si-aleatoriu>
set SMTP_USER=adresa@gmail.com
set SMTP_PASSWORD=<app-password-gmail>

# 3. Variabile opționale
set SMTP_HOST=smtp.gmail.com     # default
set SMTP_PORT=587                 # default
set TOKEN_EXPIRE_MINUTES=1440    # default 24h

# 4. Pornire
uvicorn server_v2:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: `http://localhost:8000/docs` — butonul **Authorize** permite testarea endpoint-urilor protejate.

### Android

1. Deschide folderul `android/` în Android Studio.
2. IP-ul serverului se configurează la runtime din dialogul din aplicație (default `192.168.1.132`).
3. La prima lansare: **Register** → **Login** → token salvat automat în `SharedPreferences`.
4. Toate cererile ulterioare trimit automat `Authorization: Bearer <token>` prin `AuthInterceptor`.

---

## 5. API Reference

### Autentificare (publice)

| Metodă | Endpoint | Descriere |
|--------|----------|-----------|
| `POST` | `/auth/register` | Înregistrare cont nou `{email, password}` |
| `POST` | `/auth/login` | Login form-encoded → `{access_token, token_type}` |
| `GET` | `/auth/me` | Profil utilizator curent (necesită token) |

### PhotoMatch (necesită JWT)

| Metodă | Endpoint | Descriere |
|--------|----------|-----------|
| `POST` | `/search_and_correct` | FAISS search pe vector 517-dim + referință |
| `POST` | `/apply_lut` | Aplică CLAHE + LUT pe imagine base64 |
| `GET` | `/lut/{basename}` | Descarcă LUT 3D (17³ × 3 float32) |
| `POST` | `/process` | Pipeline complet server-side (CLIP + FAISS + LUT) |
| `POST` | `/batch/process` | Până la 100 imagini într-o singură cerere |
| `POST` | `/cluster` | K-Means pe vectori 517-dim |
| `POST` | `/style/upload` | Înregistrează profil de stil din imagini |
| `POST` | `/style/vectors` | Înregistrează profil de stil din vectori |
| `POST` | `/style/process` | Corecție cu constrângere de stil |
| `POST` | `/style/search` | Retrieval cu constrângere de stil |

### BlurAPI (necesită JWT)

| Metodă | Endpoint | Parametru | Descriere |
|--------|----------|-----------|-----------|
| `POST` | `/blur-sensitive` | `?detector=gemini` | Blurare via Gemini Vision |
| `POST` | `/blur-sensitive` | `?detector=local` | Blurare via YOLOv8 local |

### Email (necesită JWT)

| Metodă | Endpoint | Descriere |
|--------|----------|-----------|
| `POST` | `/mail/send` | Trimite fotografii ca atașamente email |

### Utilitare (publice)

| Metodă | Endpoint | Descriere |
|--------|----------|-----------|
| `GET` | `/health` | Status server, vectori încărcați, device, cache-uri |
| `GET` | `/latency` | Statistici latență (mean/std/min/max per operație) |
| `GET` | `/image/raw/{basename}` | Servește imaginea RAW din dataset |
| `GET` | `/image/edited/{basename}` | Servește imaginea editată din dataset |

---

## 6. Structura proiectului

```
mvpBachelor/
├── server/
│   ├── server_v2.py              ← server principal (toate endpoint-urile)
│   ├── requirements.txt
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── database.py           ← SQLite engine + get_db()
│   │   ├── models.py             ← User model (SQLAlchemy)
│   │   ├── schemas.py            ← Pydantic: UserCreate, Token, UserResponse
│   │   ├── service.py            ← JWT, bcrypt, authenticate_user()
│   │   ├── dependencies.py       ← get_current_active_user (OAuth2)
│   │   └── router.py             ← /auth/register, /auth/login, /auth/me
│   ├── blur_api/
│   │   ├── __init__.py
│   │   ├── gemini.py             ← detect_sensitive() via Gemini Vision
│   │   └── blur.py               ← apply_blur() cu OpenCV GaussianBlur
│   ├── defect_head.pt            ← checkpoint MobileNetV3Small (4.1 MB)
│   ├── hybrid_vectors.npz        ← vectori 3499 × 517 (7.8 MB)
│   └── images/
│       ├── raw/                  ← FiveK RAW JPEGs
│       └── edited/               ← Expert-edited JPEGs
├── android/
│   └── src/main/java/com/photomatch/
│       ├── api/
│       │   ├── ApiClient.java    ← OkHttp + AuthInterceptor JWT
│       │   ├── ApiService.java   ← interfață Retrofit (toate endpoint-urile)
│       │   ├── UserCreate.java
│       │   ├── UserResponse.java
│       │   ├── TokenResponse.java
│       │   ├── MailSendResponse.java
│       │   └── ...               ← modele PhotoMatch existente
│       ├── MainActivity.java
│       ├── ProcessingActivity.java
│       ├── ResultsActivity.java
│       ├── BatchActivity.java
│       ├── ClusterActivity.java
│       ├── BurstActivity.java
│       ├── FaceGroupsActivity.java
│       └── PipelineActivity.java
└── photo_mailer/
    ├── face_db.py
    ├── scraper.py
    ├── matcher.py
    ├── mailer.py
    ├── main.py
    └── requirements.txt
```

---

> Pentru detalii despre arhitectura modelelor ML (CLIP, MobileNetV3, vectorul hibrid, LUT 3D) vezi secțiunea **Model Architecture** din [`main` branch README](https://github.com/alexandraelenadumitrescu/mvpBachelor/blob/main/README.md).
