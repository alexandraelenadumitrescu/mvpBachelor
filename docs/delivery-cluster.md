# PhotoMatch Lite — Delivery & Cluster

## Cuprins
1. [Arhitectură generală](#arhitectură-generală)
2. [Autentificare](#autentificare)
3. [Pipeline Delivery — 3 faze](#pipeline-delivery--3-faze)
   - [Faza 1 — Procesare locală](#faza-1--procesare-locală-on-device)
   - [Faza 2 — Matching pe server](#faza-2--matching-pe-server)
   - [Faza 3 — Trimitere email-uri](#faza-3--trimitere-email-uri)
4. [Cluster (diagnostic)](#cluster-diagnostic)
5. [Construirea bazei de date de fețe](#construirea-bazei-de-date-de-fețe)
6. [Algoritm de matching](#algoritm-de-matching)
7. [Harta endpoint-urilor](#harta-endpoint-urilor)
8. [Modele de date](#modele-de-date)

---

## Arhitectură generală

```mermaid
graph TB
    subgraph Android["📱 Android App"]
        UI[UI / Activities]
        MLKit[Google MLKit\nFace Detection]
        TFLite[FaceNet TFLite\n128-dim embeddings]
        UI --> MLKit
        UI --> TFLite
    end

    subgraph Server["🖥️ FastAPI Server (192.168.1.130:8000)"]
        Auth[Auth Module\nJWT]
        MatchEmb[/delivery/match-embeddings]
        SendMatched[/delivery/send-matched]
        Cluster[/cluster/faces]
        FaceDB[Face DB Builder\nTFLite + DeepFace detect]
        Matcher[Matcher\ncosine sim ≥ 0.55]
        Auth --> MatchEmb
        Auth --> SendMatched
        Auth --> Cluster
        MatchEmb --> FaceDB
        MatchEmb --> Matcher
    end

    subgraph External["🌐 Servicii externe"]
        NexusPage[nexus.html\nGitHub Pages]
        Gmail[Gmail SMTP\nsmtp.gmail.com:587]
        Inbox[alexandradumitrescu04\n@gmail.com]
    end

    Android -->|"JWT Bearer"| Server
    FaceDB -->|"HTTP scrape"| NexusPage
    SendMatched -->|"STARTTLS"| Gmail
    Gmail --> Inbox
```

---

## Autentificare

```mermaid
sequenceDiagram
    participant App as 📱 Android App
    participant API as 🖥️ FastAPI /auth/login
    participant DB as 🗄️ users.db (SQLite)

    App->>+API: POST /auth/login\nForm: username, password
    API->>+DB: SELECT user WHERE email=username
    DB-->>-API: user row (hashed password)
    API->>API: bcrypt.verify(password, hash)
    alt Credențiale valide
        API-->>App: 200 OK\n{access_token, token_type}
        Note over App: Token salvat în SharedPreferences
    else Credențiale invalide
        API-->>-App: 401 Unauthorized
    end

    Note over App: La fiecare request următor:\nAuthorization: Bearer <token>
```

---

## Pipeline Delivery — 3 faze

### Vedere de ansamblu

```mermaid
flowchart LR
    A([📷 Fotografii selectate\ndin galerie / folder]) --> B

    subgraph F1["Faza 1 — ON DEVICE"]
        B[Pentru fiecare poză\none at a time] --> C[MLKit\nFace Detection]
        C --> D[FaceNet TFLite\nEmbed 128-dim]
        D --> E[photo_index +\nembeddings list]
    end

    E --> F

    subgraph F2["Faza 2 — SERVER"]
        F[/delivery/match-embeddings\nJSON ~KB] --> G[Scrape nexus.html]
        G --> H[Build Face DB\n17 angajați]
        H --> I[Cosine similarity\nthreshold 0.55]
        I --> J[matches:\nphoto_index → email]
    end

    J --> K

    subgraph F3["Faza 3 — PER PERSOANĂ"]
        K[Grupare\nemail → photo_indices] --> L[Pentru fiecare email\nîncarcă pozele sale]
        L --> M[/delivery/send-matched\nMultipart ~MB]
        M --> N[Gmail SMTP]
        N --> O[📧 alexandradumitrescu04\n@gmail.com]
    end
```

---

### Faza 1 — Procesare locală (on-device)

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant DA as DeliveryActivity
    participant MLKit as MLKit FaceDetector
    participant TF as FaceNet TFLite

    User->>DA: Apasă "Run delivery"
    DA->>DA: setUiLoading(true)

    loop Pentru fiecare poză i din selectedUris
        DA->>DA: decodeBitmap(uri, maxSide=1024)
        Note over DA: O singură poză în memorie
        DA->>+MLKit: detect(bitmap)
        MLKit-->>-DA: List<Rect> bounding boxes
        loop Pentru fiecare față detectată
            DA->>DA: cropFace(bitmap, box, padding=20%)
            DA->>+TF: embed(faceCrop 160×160)
            Note over TF: Normalizare: (px - 127.5) / 128\nOutput: 128 floats L2-normalized
            TF-->>-DA: float[128]
        end
        DA->>DA: bitmap.recycle()
        DA->>DA: photoEmbList.add({photo_index: i,\nembeddings: [[128 floats], ...]})
    end

    DA->>DA: Lansează Faza 2
```

---

### Faza 2 — Matching pe server

```mermaid
sequenceDiagram
    participant DA as DeliveryActivity
    participant API as POST /delivery/match-embeddings
    participant Scraper as photo_mailer/scraper.py
    participant FaceDB as photo_mailer/face_db.py
    participant TFServer as FaceNet TFLite (server)

    DA->>+API: POST /delivery/match-embeddings\nJSON Body:\n{employees_url, photos:[{photo_index, embeddings:[[]]}]}
    Note over DA,API: Payload mic — doar embeddings\n91 poze × 5 fețe × 128 floats ≈ 230 KB

    API->>+Scraper: scrape_employees(employees_url)
    Scraper->>Scraper: HTTP GET nexus.html
    Scraper->>Scraper: BeautifulSoup\nparse article.speaker-card
    Scraper-->>-API: [{name, email, photo_url}] × 17

    API->>+FaceDB: build_db(employees)
    loop Pentru fiecare angajat
        FaceDB->>FaceDB: HTTP GET photo_url
        FaceDB->>FaceDB: DeepFace.extract_faces()\ndetector=opencv
        FaceDB->>+TFServer: tflite_embedder.embed(face_crop)
        Note over TFServer: Același model ca Android\nfacenet.tflite
        TFServer-->>-FaceDB: float[128] L2-normalized
        FaceDB->>FaceDB: db[email] = embedding
    end
    FaceDB-->>-API: {email: np.array[128]} × 17

    loop Pentru fiecare (photo_index, embedding) primit
        API->>API: cosine_sim = dot(embedding, db_emb)\n[L2-normalized → dot = cosine]
        API->>API: if sim ≥ 0.55 → match
    end

    API-->>-DA: 200 OK\n{matches: [{photo_index, email}],\nemployees_count: 17}
```

---

### Faza 3 — Trimitere email-uri

```mermaid
sequenceDiagram
    participant DA as DeliveryActivity\n(executor thread)
    participant API as POST /delivery/send-matched
    participant SMTP as Gmail SMTP\nsmtp.gmail.com:587
    participant Inbox as 📧 Inbox

    DA->>DA: Grupare matches:\n{email → Set<photo_index>}

    loop Pentru fiecare email (SINCRON .execute())
        DA->>DA: Încarcă doar pozele\nacelui email din selectedUris
        DA->>+API: POST /delivery/send-matched\nForm: to_email\nMultipart: photos[]
        Note over DA,API: O persoană per request\nMemorie eliberată după fiecare

        API->>API: Construiește MIMEMultipart\nSubiect: "Event photos for {to_email}"
        API->>+SMTP: STARTTLS + login\nframeshuttr@gmail.com
        SMTP-->>-API: Auth OK
        API->>SMTP: sendmail(\n  from: frameshuttr@gmail.com,\n  to: alexandradumitrescu04@gmail.com,\n  msg cu poze atașate\n)
        SMTP-->>API: Delivered

        API-->>-DA: 200 OK\n{emails_sent:1, failed:0,\ndetails:[{email, photos_count}]}
        DA->>DA: emailsSent++\nActualizează tvProgress
    end

    DA->>DA: runOnUiThread:\nAfișează "Trimise: N · Eșuate: M"
```

---

## Cluster (diagnostic)

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant CA as ClusterActivity
    participant MLKit as MLKit FaceDetector
    participant TF as FaceNet TFLite

    User->>CA: Selectează fotografii\nsau folder
    User->>CA: Apasă "Clusterizează fețe"
    CA->>CA: executor.execute(runClustering)

    loop Pentru fiecare poză (on executor thread)
        CA->>CA: decodeBitmap(uri, 1024)
        CA->>+MLKit: detect(bitmap)
        MLKit-->>-CA: List<Rect>
        loop Pentru fiecare față
            CA->>CA: cropFace + resize 112×112
            CA->>+TF: embed(crop)
            TF-->>-CA: float[128] L2-normalized
            CA->>CA: allFaces.add(\n  FaceEmbedding(uri, thumb, emb, faceIdx)\n)
        end
        CA->>CA: bitmap.recycle()
    end

    CA->>CA: FaceClusterer.cluster(allFaces,\n  threshold=0.75)
```

```mermaid
flowchart TD
    A[allFaces list\nN embeddings] --> B

    subgraph Greedy["Greedy Cosine Clustering — O(N²)"]
        B[Pentru fiecare față i\nneasignată] --> C[Crează cluster nou\ncu față i]
        C --> D[Pentru fiecare față j > i\nneasignată]
        D --> E{dot product\ni·j > 0.75?}
        E -->|Da| F[Adaugă j la cluster\nasigned[j] = true]
        E -->|Nu| D
        F --> D
        D --> G{Mai sunt j?}
        G -->|Da| D
        G -->|Nu| H[Salvează cluster\ntreci la i+1]
        H --> B
    end

    B --> I[Sortare descrescător\ndupă mărime]
    I --> J[Asignare personIndex:\n≥2 fețe → 1,2,3...\n1 față → 0 Unmatched]
    J --> K[runOnUiThread: showResults]
```

```mermaid
flowchart LR
    K[clusters list] --> L[tvSummary:\nN fețe · M persoane · K unice]

    subgraph Display["UI — ScrollView"]
        L --> M
        loop Pentru fiecare cluster
            M[Title: Persoana X / Față unică] --> N[HorizontalScrollView\ncu ImageView-uri 100×100]
            N --> M
        end
    end
```

---

## Construirea bazei de date de fețe

```mermaid
flowchart TD
    A([employees_url\nnexus.html]) --> B[scraper.py\nBeautifulSoup parse]
    B --> C{Format detectat}
    C -->|article.speaker-card| D[Extrage:\nname, email,\ndata-email,\nimg.src]
    C -->|.employee-card| E[Extrage:\nname, p email,\nimg.src]
    D --> F
    E --> F

    F[employees list] --> G

    subgraph BuildDB["face_db.py — build_db()"]
        G[Pentru fiecare angajat] --> H[HTTP GET photo_url\ntimeout=10s]
        H --> I{Succes?}
        I -->|Nu| J[skip — log ✗]
        I -->|Da| K[Salvează în fișier temp .jpg]
        K --> L[DeepFace.extract_faces\ndetector=opencv\nenforce_detection=False]
        L --> M[face_arr float64 0-1\n→ PIL Image uint8]
        M --> N[tflite_embedder.embed\nResize 160×160\nNormalizare 127.5/128\nTFLite invoke]
        N --> O[L2-normalize\nfloat32 128-dim]
        O --> P[db[email] = embedding]
        P --> G
    end

    P --> Q[(Face DB\n{email: np.array128})]
```

---

## Algoritm de matching

```mermaid
flowchart TD
    A([embedding primit\nfloat32 128-dim\nde la telefon]) --> B[L2-normalize\ndacă norm > 0]

    B --> C[Pentru fiecare email\ndin Face DB]

    subgraph Match["_best_match per embedding"]
        C --> D["sim = dot(embedding, db_emb)\n= cosine similarity\n(ambele L2-normalized)"]
        D --> E{sim > best_sim?}
        E -->|Da| F[best_sim = sim\nbest_email = email]
        F --> C
        E -->|Nu| C
        C --> G{Toți angajații\nprocesați?}
    end

    G --> H{best_sim ≥ 0.55?}
    H -->|Da| I{email deja\nmatch în poza asta?}
    I -->|Nu| J[✓ Adaugă match\nPhotoMatchItem]
    I -->|Da| K[Skip — evită\nduplicat per poză]
    H -->|Nu| L[✗ Sub threshold\nignorат]
```

---

## Harta endpoint-urilor

```mermaid
graph LR
    subgraph Auth["🔑 Autentificare"]
        L1[POST /auth/login]
        L2[POST /auth/register]
    end

    subgraph DeliveryV2["📦 Delivery v2 — embedding-based"]
        D1[POST /delivery/match-embeddings\nJSON body\nReturns: matches list]
        D2[POST /delivery/send-matched\nMultipart per persoană\nReturns: emails_sent]
    end

    subgraph DeliveryV1["📦 Delivery v1 — legacy"]
        D3[POST /delivery/run\nMultipart toate pozele\nFull server-side pipeline]
    end

    subgraph Cluster["🔬 Cluster diagnostic"]
        C1[POST /cluster/faces\nMultipart\nReturns: clusters cu base64]
    end

    subgraph Misc["🔧 Diverse"]
        M1[GET /mock-employees\nHTML demo angajați]
        M2[POST /blur-sensitive\nBlurare imagini]
        M3[POST /mail/send\nTrimiteri directe]
    end

    User((👤 User)) -->|JWT| D1
    User -->|JWT| D2
    User -->|JWT| D3
    User -->|JWT| C1
    User --> L1
    User --> L2
```

---

## Modele de date

### Request — match-embeddings

```mermaid
classDiagram
    class MatchEmbeddingsRequest {
        +String employees_url
        +List~PhotoEmbeddingsItem~ photos
    }
    class PhotoEmbeddingsItem {
        +int photo_index
        +List~List~Float~~ embeddings
    }
    MatchEmbeddingsRequest "1" --> "*" PhotoEmbeddingsItem
```

### Response — match-embeddings

```mermaid
classDiagram
    class MatchEmbeddingsResponse {
        +List~PhotoMatchItem~ matches
        +int employees_count
    }
    class PhotoMatchItem {
        +int photo_index
        +String email
    }
    MatchEmbeddingsResponse "1" --> "*" PhotoMatchItem
```

### Face DB (server in-memory)

```mermaid
classDiagram
    class FaceDB {
        +Dict~str_ndarray128~ db
        +build_db(employees) Dict
    }
    class Employee {
        +String name
        +String email
        +String photo_url
    }
    class Embedding {
        +float32[128] vector
        +L2_normalized bool
    }
    FaceDB "1" --> "*" Employee : scrape
    FaceDB "1" --> "*" Embedding : stores
```

### Cluster (on-device Android)

```mermaid
classDiagram
    class FaceCluster {
        +int personIndex
        +List~FaceEmbedding~ faces
    }
    class FaceEmbedding {
        +String photoUri
        +Bitmap crop_112x112
        +float[128] embedding
        +int faceIndex
    }
    FaceCluster "1" --> "*" FaceEmbedding
```

---

## Fluxul complet end-to-end

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant App as 📱 Android App
    participant Server as 🖥️ FastAPI Server
    participant NexusPage as 🌐 nexus.html
    participant Gmail as 📧 Gmail SMTP
    participant Inbox as 📥 Inbox

    User->>App: Login (a@a.com / string)
    App->>Server: POST /auth/login
    Server-->>App: JWT token

    User->>App: Selectează folder cu poze\n(N fotografii)
    Note over App: FolderPickerHelper\nDocumentFile.listFiles()

    User->>App: Apasă Run Delivery

    rect rgb(200, 230, 200)
        Note over App: FAZA 1 — ON DEVICE
        loop Fiecare poză (one at a time)
            App->>App: MLKit detect faces
            App->>App: FaceNet TFLite embed\n128-dim L2-normalized
            App->>App: bitmap.recycle()
        end
    end

    rect rgb(200, 200, 230)
        Note over App,Server: FAZA 2 — SERVER MATCHING (~KB payload)
        App->>Server: POST /delivery/match-embeddings\n{employees_url, photos:[{idx, embs}]}
        Server->>NexusPage: HTTP GET nexus.html
        NexusPage-->>Server: 17 angajați (name, email, photo_url)
        loop Fiecare angajat
            Server->>Server: download photo\nDeepFace detect\nTFLite embed
        end
        Server->>Server: cosine sim ≥ 0.55\npentru fiecare embedding primit
        Server-->>App: {matches:[{photo_index, email}]}
    end

    rect rgb(230, 200, 200)
        Note over App,Inbox: FAZA 3 — EMAIL (per persoană)
        loop Fiecare email SINCRON
            App->>App: Încarcă pozele persoanei
            App->>Server: POST /delivery/send-matched\nForm: to_email\nFiles: photos[]
            Server->>Gmail: STARTTLS\nsendmail
            Gmail->>Inbox: "Event photos for {email}"\n+ poze atașate
            Server-->>App: {emails_sent:1}
            App->>App: Eliberează memoria
        end
    end

    App->>User: "Trimise: N · Eșuate: M"
```
