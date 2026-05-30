# Fluxuri detaliate — Client Android & Server FastAPI

Fiecare secțiune arată: **ce intră → ce se procesează → unde pleacă → ce se întoarce → ce se afișează**.

---

## Cuprins

**Client Android**
1. [Login / Register](#1-login--register)
2. [Corecție fotografie unică](#2-corecție-fotografie-unică)
3. [Batch processing](#3-batch-processing)
4. [Burst mode + best shot](#4-burst-mode--best-shot)
5. [Clustering fotografii](#5-clustering-fotografii)
6. [Stil personalizat](#6-stil-personalizat)
7. [Grupare după fețe](#7-grupare-după-fețe)
8. [Favorite](#8-favorite)
9. [Pipeline vizualizare pas-cu-pas](#9-pipeline-vizualizare-pas-cu-pas)
10. [Blur conținut sensibil](#10-blur-conținut-sensibil)
11. [Trimitere email fotografii](#11-trimitere-email-fotografii)

**Server FastAPI**
12. [/auth/register](#12-authregister)
13. [/auth/login](#13-authlogin)
14. [/search_and_correct](#14-search_and_correct)
15. [/blur-sensitive — calea Gemini](#15-blur-sensitive--calea-gemini)
16. [/blur-sensitive — calea locală YOLOv8](#16-blur-sensitive--calea-locală-yolov8)
17. [/mail/send](#17-mailsend)
18. [/batch/process](#18-batchprocess)
19. [/cluster](#19-cluster)
20. [/lut/{basename}](#20-lutbasename)
21. [/style/*](#21-style)

**Explicații detaliate**
22. [Cum funcționează blur — detaliu complet](#22-cum-funcționează-blur--detaliu-complet)
23. [Cum funcționează mail — detaliu complet](#23-cum-funcționează-mail--detaliu-complet)

---

## CLIENT ANDROID — ACTIVITĂȚI

---

### 1. Login / Register

**Activitate:** `LoginActivity`

```mermaid
flowchart TD
    A([Utilizator deschide aplicația]) --> B{Token JWT\nexistă în\nSharedPreferences?}
    B -->|Da| Z([MainActivity — direct])
    B -->|Nu| C[Afișează ecran Login]

    C --> D[Utilizator introduce\nemail + parolă]
    D --> E{Register\nsau Login?}

    E -->|Register| F["POST /auth/register\n{email, password}"]
    F --> F1{Răspuns}
    F1 -->|201 UserResponse| F2[Cont creat\n→ mergi la Login]
    F1 -->|400 email există| F3[Toast: 'Email deja înregistrat']

    E -->|Login| G["POST /auth/login\nform-encoded:\nusername=email\npassword=pass"]
    G --> G1{Răspuns}
    G1 -->|200 TokenResponse| G2["ApiClient.saveToken(access_token)\n→ MainActivity"]
    G1 -->|401 invalid| G3[Toast: 'Email sau parolă incorectă']
```

**Ce pleacă spre server:**
- Register: JSON `{"email": "...", "password": "..."}`
- Login: form-data `username=...&password=...` (OAuth2 standard)

**Ce se întoarce:**
- Register: `{"id": 1, "email": "...", "is_active": true, "created_at": "..."}`
- Login: `{"access_token": "eyJ...", "token_type": "bearer"}`

---

### 2. Corecție fotografie unică

**Activități:** `ProcessingActivity` → `ResultsActivity` → `CompareActivity` / `DetailActivity`

```mermaid
flowchart TD
    A([Utilizator selectează\nfotografie din galerie/cameră]) --> B

    subgraph ONDEV["📱 On-Device — imaginea NU pleacă din telefon"]
        B["CLIPEncoder.java\nTFLite inference\n→ 512-dim embedding"]
        B --> C["DefectDetector.java\nMobileNetV3 TFLite\n→ 5 scoruri: blur, noise,\noverexposure, underexposure,\ncompression"]
        C --> D["HybridVectorBuilder.java\nnormalizare separată\n× 1.0 CLIP + × 5.0 defect\n→ 517-dim vector L2-norm"]
    end

    D --> E["POST /search_and_correct\n{vector: [517 float-uri]}\nAuthorization: Bearer JWT"]

    subgraph SERVER["🖥️ Server"]
        E --> F["FAISS IndexFlatIP\nsearch top-10"]
        F --> G["_aesthetic_rerank()\ncombinare similarity + aesthetic_score\n→ cel mai bun basename"]
        G --> H["Returnează: retrieved, similarity,\nlut_cached, raw_b64, edited_b64"]
    end

    H --> I{LUT cached\nlocal?}
    I -->|Nu| J["GET /lut/{basename}\n→ 17³×3 float32 bytes base64"]
    J --> K["LutCache.java\nsalvează LUT pe disc"]
    I -->|Da| K

    K --> L["ImageCorrector.java\nCLAHE pe canalul L\n+ aplicare LUT la strength 0.2"]
    L --> M(["ResultsActivity\nafișează: Original | CLAHE | Final\n+ defect scores\n+ imagine referință RAW/Edited\n+ scor similaritate"])
```

**Ce pleacă spre server:** vectorul de 517 float-uri (nu imaginea)

**Ce se întoarce:** basename-ul imaginii de referință + LUT-ul ca bytes + thumbnail-uri base64

---

### 3. Batch processing

**Activități:** `BatchActivity` → `BatchResultsActivity`

```mermaid
flowchart TD
    A([Utilizator selectează\npână la 100 imagini]) --> B

    subgraph LOOP["Repetat pentru fiecare imagine — on-device"]
        B["CLIP + MobileNetV3\n→ 517-dim vector"]
    end

    B --> C["POST /batch/process\nMultipart: files[]\npână la 100 fișiere JPEG"]

    subgraph SERVER_B["Server — procesare secvențială"]
        C --> D["Pentru fiecare imagine:\nCLIP + MobileNetV3 server-side\n→ FAISS → CLAHE + LUT"]
        D --> E["Lista BatchResult:\nindex, retrieved, similarity,\ncorrected_b64, aesthetic_score"]
    end

    E --> F(["BatchResultsActivity\nGrid cu toate imaginile corectate\n+ statistici per imagine"])
```

> **Notă:** `/batch/process` rulează inferența CLIP pe server (nu trimite vectorii), spre deosebire de `/search_and_correct` unde clientul trimite vectorul calculat local.

---

### 4. Burst mode + best shot

**Activități:** `BurstActivity` → `BurstResultsActivity`

```mermaid
flowchart TD
    A([Utilizator face o serie\nde fotografii rapide]) --> B

    subgraph ONDEV_B["On-Device"]
        B["CLIP + MobileNetV3\npentru fiecare cadru\n→ vectori 517-dim"]
        B --> C["BlurDetector.java\nvarianta Laplacian\nfiltru cadre neclare"]
    end

    C --> D["POST /cluster\n{vectors: [[517...], [517...], ...]}\nn_clusters: null → auto-k"]

    subgraph SERVER_C["Server"]
        D --> E["_auto_k() — metoda cotului\nK-Means (k între 2 și 8)\nsilhouette score"]
        E --> F["Returnează grupuri:\n{cluster_id, indices[]}"]
    end

    F --> G["BurstClusterer.java\npentru fiecare grup:\nalege cadrul cu cel mai\nmare scor estetic local"]
    G --> H(["BurstResultsActivity\nAfișează best-shot per grup\n+ toate cadrele grupate"])
```

---

### 5. Clustering fotografii

**Activități:** `ClusterActivity` → `ClusterResultsActivity`

```mermaid
flowchart TD
    A([Utilizator selectează\no colecție de fotografii]) --> B["CLIP + MobileNetV3\npentru fiecare imagine\n→ vectori 517-dim"]
    B --> C["POST /cluster\n{vectors, n_clusters: N sau null}"]

    subgraph SERVER_CL["Server"]
        C --> D{n_clusters\nspecificat?}
        D -->|Nu| E["_auto_k(X)\nmetoda cotului inertie\npână la 8 clustere"]
        D -->|Da| F["k specificat"]
        E --> G["KMeans fit_predict\nsklearn"]
        F --> G
        G --> H["silhouette_score\npentru evaluare calitate"]
        H --> I["Returnează:\n{clusters, n_clusters,\nsilhouette_score}"]
    end

    I --> J(["ClusterResultsActivity\nAfișează grupuri vizual\n+ scor silhouette pe titlu"])
```

---

### 6. Stil personalizat

**Activitate:** `StyleSetupActivity` → orice activitate de corecție cu stil

```mermaid
flowchart TD
    A([Utilizator selectează\n1-20 fotografii de referință\nstil personal]) --> B

    subgraph ONDEV_S["On-Device"]
        B["CLIP + MobileNetV3\npentru fiecare referință\n→ vectori 517-dim"]
    end

    B --> C["POST /style/vectors\n{vectors: [[517...], ...],\nsession_id: null}"]
    C --> D["Răspuns:\n{session_id: 'uuid', vectors_stored: N}"]
    D --> E["session_id salvat local"]

    E --> F([Utilizator face o fotografie nouă])
    F --> G["CLIP + MobileNetV3\n→ vector 517-dim"]
    G --> H["POST /style/search\n{vector, session_id}"]

    subgraph SERVER_S["Server"]
        H --> I["Centroid + raza\ndin vectorii de stil"]
        I --> J["FAISS top-51\nfiltru regiune stil\n(threshold dinamic)"]
        J --> K["Semantic guard:\nCLIP-only cosine ≥ 0.55"]
        K --> L["Returnează: retrieved, similarity,\nstyle_matched, style_fallback,\nlut_cached"]
    end

    L --> M["GET /lut/{basename}\ndacă nu e în cache"]
    M --> N(["Afișează rezultat cu indicator\nstyle_matched / style_fallback"])
```

---

### 7. Grupare după fețe

**Activitate:** `FaceGroupsActivity`

```mermaid
flowchart TD
    A([Utilizator selectează\no colecție de fotografii]) --> B

    subgraph ONDEV_F["100% On-Device — niciun apel la server"]
        B --> C["FaceDetectorHelper.java\nMLKit Face Detection\npentru fiecare fotografie"]
        C --> D["FaceEmbedder.java\nFaceNet 512-dim\nembedding per față detectată"]
        D --> E["FaceClusterer.java\nCosine similarity\nîntre toți vectorii FaceNet\nDBSCAN / threshold clustering"]
    end

    E --> F(["FaceGroupsActivity\nAfișează grupuri:\nPersoana 1 (N foto),\nPersoana 2 (M foto), ..."])
```

> **Important:** această funcționalitate este complet on-device — nicio imagine și niciun vector nu pleacă din telefon.

---

### 8. Favorite

**Activități:** `FavoritesActivity` → `FavoriteDetailActivity`

```mermaid
flowchart TD
    A([Din ResultsActivity\nutilizatorul apasă ♥]) --> B["Room INSERT\nFavoritePhoto:\nimagePath, retrievedBasename,\nsimilarity, timestamp"]

    B --> C(["FavoritesActivity\nRoom SELECT * ORDER BY timestamp\nAfișează grid cu toate favoritele"])
    C --> D([Utilizator apasă pe o fotografie])
    D --> E["Room SELECT by id"]
    E --> F(["FavoriteDetailActivity\nAfișează detalii:\nscor similaritate, basename,\ndata salvării"])
    F --> G{Utilizator dorește\nre-corecție?}
    G -->|Da| H["Reia fluxul de la\nCorecție unică cu aceeași imagine"]
    G -->|Șterge| I["Room DELETE\nDispare din favorite"]
```

---

### 9. Pipeline vizualizare pas-cu-pas

**Activitate:** `PipelineActivity`

```mermaid
flowchart TD
    A([Utilizator selectează\n'Pipeline' din MainActivity]) --> B["Selectează o fotografie"]
    B --> C

    subgraph STEPS["Pași vizualizați interactiv"]
        C["Pas 1: Original\nafișat în RecyclerView"]
        C --> D["Pas 2: CLIP encode on-device\nafișează vectorul parțial"]
        D --> E["Pas 3: MobileNetV3 on-device\nafișează scorurile defect"]
        E --> F["Pas 4: POST /search_and_correct\nafișează imaginea de referință găsită"]
        F --> G["Pas 5: CLAHE aplicat\nafișează rezultat intermediar"]
        G --> H["Pas 6: LUT aplicat\nafișează rezultatul final"]
    end

    H --> I(["PipelineResultsBottomSheet\ncomparație side-by-side\nOriginal vs Final"])
```

---

### 10. Blur conținut sensibil

**Activitate:** `BlurActivity` *(nou adăugată)*

```mermaid
flowchart TD
    A([Utilizator selectează o fotografie\ncare conține conținut sensibil]) --> B["Afișează imaginea originală\nîn ImageView stânga"]
    B --> C{Utilizator alege\ndetectorul}

    C -->|Gemini| D["POST /blur-sensitive?detector=gemini\nMultipart: file=<JPEG bytes>\nAuthorization: Bearer JWT"]
    C -->|Local YOLOv8| E["POST /blur-sensitive?detector=local\nMultipart: file=<JPEG bytes>\nAuthorization: Bearer JWT"]

    D --> F["Server Gemini path\n(detalii secțiunea 22)"]
    E --> G["Server YOLOv8 path\n(detalii secțiunea 22)"]

    F --> H["Răspuns: JPEG bytes\n(nu JSON — imagine directă)"]
    G --> H

    H --> I["Bitmap.decodeByteArray(body.bytes())"]
    I --> J(["Afișează imaginea blurată\nîn ImageView dreapta\nbefore/after side-by-side"])

    J --> K{Utilizator\nsalvează?}
    K -->|Da| L["Salvează JPEG în galerie"]
    K -->|Nu| M([Revine la selecție])
```

---

### 11. Trimitere email fotografii

**Activitate:** `DeliveryActivity` *(nou adăugată)*

```mermaid
flowchart TD
    A([Utilizator apasă 'Trimite email'\ndin aplicație]) --> B["Selectează fotografiile\n(multi-image picker)"]
    B --> C["Introduce adresa destinatar\n+ subiect + mesaj optional"]
    C --> D["POST /mail/send\nMultipart form-data:\nto_email, subject, message\n+ files[] JPEG\nAuthorization: Bearer JWT"]

    D --> E["Server procesează\n(detalii secțiunea 23)"]

    E --> F{Răspuns}
    F -->|200 MailSendResponse| G["Toast:\n'Email trimis către email@...'\nN fotografii atașate"]
    F -->|5xx eroare SMTP| H["Toast: eroare server\n(credențiale SMTP greșite\nsau conexiune eșuată)"]
    F -->|401 unauthorized| I["Token expirat\n→ redirect la Login"]
```

---

## SERVER FASTAPI — ENDPOINT-URI

---

### 12. /auth/register

```mermaid
flowchart TD
    IN["POST /auth/register\nJSON: {email, password}"] --> A

    A["Pydantic validare\nEmailStr format"] --> B{Email există\nîn users.db?}
    B -->|Da| ERR1["HTTP 400\n'Email deja înregistrat'"]
    B -->|Nu| C["hash_password(password)\nbcrypt cost factor 12\n→ hashed_password string"]
    C --> D["INSERT INTO users\n(email, hashed_password,\ncreated_at=now, is_active=True)"]
    D --> E["db.commit() + db.refresh()"]
    E --> OK["HTTP 201\nUserResponse:\n{id, email, is_active, created_at}"]
```

---

### 13. /auth/login

```mermaid
flowchart TD
    IN["POST /auth/login\nform-data: username + password"] --> A

    A["OAuth2PasswordRequestForm\nautomatically parsed by FastAPI"] --> B["authenticate_user(db, email, pass)"]
    B --> C["SELECT * FROM users\nWHERE email = :email"]
    C --> D{User\ngăsit?}
    D -->|Nu| ERR1["HTTP 401\n'Email sau parolă incorectă'"]
    D -->|Da| E["verify_password(plain, hashed)\nbcrypt compare"]
    E --> F{Parolă\ncorectă?}
    F -->|Nu| ERR1
    F -->|Da| G["create_access_token({sub: email})\nHS256 sign cu JWT_SECRET_KEY\nexp = now + TOKEN_EXPIRE_MINUTES"]
    G --> OK["HTTP 200\nToken:\n{access_token: 'eyJ...', token_type: 'bearer'}"]
```

---

### 14. /search_and_correct

```mermaid
flowchart TD
    IN["POST /search_and_correct\nJSON: {vector: [517 floats]}\nBearer JWT"] --> AUTH

    AUTH["get_current_active_user()\ndecode JWT → email → SELECT user"] --> VAL
    AUTH -->|token invalid| E401["HTTP 401"]

    VAL{Dimensiune\nvector = 517?} -->|Nu| E400["HTTP 400\n'Expected 517 dims'"]
    VAL -->|Da| NORM["L2 normalize query\nquery / ||query||"]

    NORM --> FAISS["FAISS IndexFlatIP\nindex.search(query, max(top_k,10)+1)\nDot product pe 3499 vectori"]
    FAISS --> CAND["Lista candidates:\n[(basename, similarity), ...]"]

    CAND --> RERANK["_aesthetic_rerank(candidates, aesthetic_weight)\nw_sim × similarity + w_aes × aesthetic_score\npentru fiecare candidat"]
    RERANK --> BEST["best_name, best_sim, best_aes\n= candidatul cu scor combinat maxim"]

    BEST --> IMGS{include_images\n= True?}
    IMGS -->|Da| LOAD["Încarcă raw + edited\ndin images/ → base64 JPEG"]
    IMGS -->|Nu| SKIP["raw_b64 = edited_b64 = ''"]

    LOAD --> OK["HTTP 200\n{retrieved, similarity,\nraw_b64, edited_b64,\nlut_cached, match_aesthetic_score}"]
    SKIP --> OK
```

---

### 15. /blur-sensitive — calea Gemini

```mermaid
flowchart TD
    IN["POST /blur-sensitive?detector=gemini\nMultipart: file=JPEG\nBearer JWT"] --> AUTH2["JWT check"]
    AUTH2 --> VAL2{content_type\nstartswith image/?}
    VAL2 -->|Nu| E400B["HTTP 400 'File must be an image'"]
    VAL2 -->|Da| READ["image_bytes = await file.read()"]

    READ --> GEM["gemini.detect_sensitive(image_bytes)"]

    subgraph GEMINI_FLOW["blur_api/gemini.py"]
        GEM --> PIL["Image.open(BytesIO).convert('RGB')\nconversie PIL pentru Gemini"]
        PIL --> PROMPT["_model.generate_content([_PROMPT, pil_image])\nPrompt: 'Identify all sensitive regions...\nReturn ONLY JSON array...'"]
        PROMPT --> RESP["response.text — poate conține:\n• JSON pur: [{label, x, y, w, h}]\n• JSON în ```json ... ``` (markdown fence)"]
        RESP --> PARSE["_parse_regions(text)\nre.sub strip markdown\njson.loads()\nvalidare isinstance(list)"]
        PARSE --> REGIONS["Lista regions:\n[{label: 'screen', x:120, y:45, w:300, h:200}]"]
    end

    REGIONS --> BLUR_F["apply_blur(image_bytes, regions)"]

    subgraph BLUR_FLOW["blur_api/blur.py"]
        BLUR_F --> DECODE["cv2.imdecode(np.frombuffer(bytes))\n→ numpy BGR array H×W×3"]
        DECODE --> LOOP["Pentru fiecare region:\nx1,y1 = clamp(x,y) la [0, w/h]\nx2,y2 = clamp(x+w, y+h) la [0, w/h]"]
        LOOP --> GBLUR["img[y1:y2, x1:x2] =\ncv2.GaussianBlur(\n  img[y1:y2, x1:x2],\n  (51, 51),  ← kernel 51×51\n  0           ← sigma automat\n)"]
        GBLUR --> ENCODE["cv2.imencode('.jpg', img,\n[IMWRITE_JPEG_QUALITY, 90])"]
    end

    ENCODE --> OK2["HTTP 200\ncontent-type: image/jpeg\nbody: JPEG bytes blurat"]
```

---

### 16. /blur-sensitive — calea locală YOLOv8

```mermaid
flowchart TD
    IN2["POST /blur-sensitive?detector=local\nMultipart: file=JPEG\nBearer JWT"] --> AUTH3["JWT check"] --> READ2["image_bytes = await file.read()"]

    READ2 --> LOCAL["local_detector.detect_sensitive(image_bytes)"]

    subgraph LOCAL_FLOW["blur_api/local_detector.py"]
        LOCAL --> NP["np.array(Image.open(BytesIO).convert('RGB'))\n→ RGB array pentru YOLOv8"]
        NP --> YOLO["_model(img, verbose=False)\nYOLOv8 nano (6 MB)\ninferență locală, fără API extern"]
        YOLO --> BOXES["results[0].boxes\nfiecare box are:\ncls (clasa COCO)\nconf (confidence)\nxyxy (coordonate)"]
        BOXES --> FILTER["Filtru cls_id:\n62 → 'screen' (TV)\n63 → 'screen' (laptop)\n67 → 'screen' (phone)\n73 → 'document' (carte)\nIgnorat: persoane, mașini etc."]
        FILTER --> CONF["Filtru conf ≥ 0.4\n(echilibru precizie/recall)"]
        CONF --> REGIONS2["Lista regions:\n[{label, x, y, w, h}]"]
    end

    REGIONS2 --> BLUR_F2["apply_blur(image_bytes, regions)\n(același cod ca Gemini)"]
    BLUR_F2 --> OK3["HTTP 200\nimage/jpeg blurat"]
```

**Diferența față de Gemini:**

| Aspect | Gemini | YOLOv8 local |
|--------|--------|--------------|
| Viteza | ~1000–1500 ms | ~80–100 ms |
| Conexiune externă | Da (Google API) | Nu (totul local) |
| Cost | Per API call | Gratuit |
| Categorii detectate | Orice obiect sensibil semantic | Numai clase COCO mapate |
| Detectează badge-uri/ID | Da | Nu (absent din COCO) |

---

### 17. /mail/send

```mermaid
flowchart TD
    IN3["POST /mail/send\nMultipart form-data:\nto_email (text)\nsubject (text)\nmessage (text)\nfiles[] (JPEG)\nBearer JWT"] --> AUTH4["get_current_active_user()\n→ current_user.email"]

    AUTH4 --> ENV["Citire credențiale din env:\nSMTP_HOST (default: smtp.gmail.com)\nSMTP_PORT (default: 587)\nSMTP_USER\nSMTP_PASSWORD"]

    ENV --> MSG["MIMEMultipart()\nmsg['From'] = SMTP_USER\nmsg['To'] = to_email\nmsg['Subject'] = subject"]

    MSG --> BODY["MIMEText(message or\n'Fotografii trimise de {user.email}')\nattach la msg"]

    BODY --> ATTACH["Pentru fiecare fișier din files[]:\n  data = await f.read()\n  img_part = MIMEImage(data, name=f.filename)\n  Content-Disposition: attachment\n  msg.attach(img_part)"]

    ATTACH --> SMTP_CONN["with smtplib.SMTP(host, port) as server:\n  server.ehlo()\n  server.starttls()  ← criptare TLS\n  server.login(user, password)\n  server.sendmail(from, to, msg.as_string())"]

    SMTP_CONN --> OK4["HTTP 200\n{sent: true,\nto: 'dest@example.com',\nphotos_attached: N}"]

    SMTP_CONN -->|Eroare SMTP| ERR_SMTP["HTTP 500\n(credențiale greșite / server indisponibil)"]
```

---

### 18. /batch/process

```mermaid
flowchart TD
    IN4["POST /batch/process\nMultipart: files[] (max 100 JPEG)\nBearer JWT"] --> CHECK{len(files) > 100?}
    CHECK -->|Da| E400C["HTTP 400 'Maximum 100 images'"]
    CHECK -->|Nu| LOOP2["Pentru fiecare fișier:"]

    subgraph BATCH_LOOP["Procesare per imagine (server-side inference)"]
        LOOP2 --> READ3["PIL.open → RGB → float32 array"]
        READ3 --> VEC2["get_hybrid_vector(pil_img)\nCLIP ViT-B-32 + MobileNetV3\n→ 517-dim vector"]
        VEC2 --> RET2["retrieve_similar(clip_vec, defect_vec, top_k=10)"]
        RET2 --> RERANK2["_aesthetic_rerank() → best match"]
        RERANK2 --> CORR2["correct_clahe() + apply_lut_moderated()\n→ CLAHE + LUT blend 0.2"]
        CORR2 --> B64["np_to_base64() → JPEG base64"]
    end

    B64 --> RESP2["HTTP 200 BatchResponse:\n{results: [BatchResult...],\nprocessed: N, failed: M}"]
```

---

### 19. /cluster

```mermaid
flowchart TD
    IN5["POST /cluster\nJSON: {vectors: [[517...], ...], n_clusters: null}\nBearer JWT"] --> CHECK2{N > 500\nsau N < 2?}
    CHECK2 -->|Da| E400D["HTTP 400"]
    CHECK2 -->|Nu| NORM2["L2 normalize\nX /= ||X||"]

    NORM2 --> K_DEC{n_clusters\nspecificat?}
    K_DEC -->|Da| USE_K["k = n_clusters"]
    K_DEC -->|Nu| AUTO["_auto_k(X, max_k=8)\nMetoda cotului:\nk unde scăderea inerției < 20%\ndin scăderea totală"]

    USE_K --> KMEANS["KMeans(n_clusters=k, n_init='auto')\nsklearn.fit_predict(X)"]
    AUTO --> KMEANS

    KMEANS --> SIL["silhouette_score(X, labels)\n(omis dacă k = n_samples)"]
    SIL --> OK5["HTTP 200\n{clusters: [{cluster_id, indices[]}, ...],\nn_clusters: k,\nsilhouette_score: 0.87}"]
```

---

### 20. /lut/{basename}

```mermaid
flowchart TD
    IN6["GET /lut/{basename}\nBearer JWT"] --> CLEAN["Path(basename).name\n(sanitizare path traversal)"]
    CLEAN --> CACHE{basename\nîn lut_cache?}

    CACHE -->|Da| SER["Serializam LUT existent"]
    CACHE -->|Nu| CHECK3{raw + edited\nexistă pe disc?}
    CHECK3 -->|Nu| E404["HTTP 404"]
    CHECK3 -->|Da| COMPUTE["raw = _load_image(raw_path)\nedited = _load_image(edited_path)"]

    COMPUTE --> LUT_EXTRACT["extract_colour_lut(raw, edited)\n1. Eșantionare 50.000 pixeli aleatori\n2. LinearNDInterpolator R,G,B separat\n3. Grilă 17×17×17 în spațiul [0,1]³\n4. Interpolare → LUT (17,17,17,3) float32"]
    LUT_EXTRACT --> STORE["lut_cache[basename] = lut"]
    STORE --> SER

    SER --> B64_LUT["lut.flatten().astype(float32).tobytes()\nbase64.b64encode()"]
    B64_LUT --> OK6["HTTP 200\n{lut_b64: '...', lut_size: 17}\n~58 KB per LUT"]
```

---

### 21. /style/*

```mermaid
flowchart TD
    subgraph SETUP["Configurare stil — POST /style/vectors"]
        S1["JSON: {vectors: [[517...], ...], session_id: null}"] --> S2["Validare dimensiuni"]
        S2 --> S3["session_id = uuid4() dacă null\nstyle_profiles[session_id] = np.array(vectors)"]
        S3 --> S4["HTTP 200: {session_id, vectors_stored}"]
    end

    subgraph SEARCH["Căutare cu stil — POST /style/search"]
        SS1["JSON: {vector, session_id}"] --> SS2["style_vecs = style_profiles[session_id]"]
        SS2 --> SS3["FAISS top-51 rezultate\npentru query vector"]
        SS3 --> SS4["_style_constrained_retrieve:\n1. centroid = mean(style_vecs), normalize\n2. radius = 1 - min(sims_to_centroid)\n3. threshold = 1 - radius × 1.2\n4. filtru: weighted_matrix[i] @ centroid ≥ threshold"]
        SS4 --> SS5["Semantic guard:\nCLIP-only cosine(query[:512], ref[:512]) ≥ 0.55\nAltfel: fallback la global top-1"]
        SS5 --> SS6["HTTP 200:\n{retrieved, similarity,\nstyle_matched, style_fallback, lut_cached}"]
    end
```

---

## EXPLICAȚII DETALIATE

---

### 22. Cum funcționează blur — detaliu complet

#### Flux end-to-end cu Gemini

```mermaid
sequenceDiagram
    actor U as Utilizator
    participant A as Android
    participant S as Server FastAPI
    participant G as Gemini Vision API

    U->>A: Selectează imagine cu ecran/document vizibil
    A->>A: Citește bytes JPEG din Uri
    A->>S: POST /blur-sensitive?detector=gemini\nAuthorization: Bearer JWT\nContent-Type: multipart/form-data\n[JPEG bytes]

    S->>S: JWT decode → utilizator valid
    S->>S: image_bytes = await file.read()
    S->>S: PIL.Image.open(BytesIO(image_bytes)).convert("RGB")

    S->>G: generate_content([PROMPT, pil_image])\nPrompt trimis: "Identify all sensitive regions...\nReturn ONLY a JSON array with absolute pixel coordinates"

    Note over G: Gemini analizează semantic imaginea:<br/>înțelege context (TV decorativă vs ecran cu date,<br/>carte vs document cu date sensibile)

    G-->>S: response.text = '[{"label":"screen","x":120,"y":45,"w":300,"h":200}]'\n(sau învelit în ```json ... ```)

    S->>S: _parse_regions(text):\n1. re.sub strip markdown fences\n2. json.loads()\n3. isinstance(list) check

    S->>S: apply_blur(image_bytes, regions):\npentru fiecare {x,y,w,h}:\n  x1,y1 = max(0, x), max(0, y)\n  x2,y2 = min(W, x+w), min(H, y+h)\n  img[y1:y2, x1:x2] = GaussianBlur(kernel=51×51)

    Note over S: Kernel 51×51 = blur puternic,<br/>conținutul devine nerecognoscibil

    S->>S: cv2.imencode(".jpg", img, quality=90)
    S-->>A: HTTP 200 Content-Type: image/jpeg\n[JPEG bytes — imaginea blurată]

    A->>A: Bitmap.decodeByteArray(body.bytes())
    A->>U: Afișează imaginea blurată side-by-side cu originalul
```

#### Flux end-to-end cu YOLOv8 local

```mermaid
sequenceDiagram
    actor U as Utilizator
    participant A as Android
    participant S as Server FastAPI

    U->>A: Selectează imagine
    A->>S: POST /blur-sensitive?detector=local\n[JPEG bytes]

    S->>S: np.array(Image.open(BytesIO).convert("RGB"))

    Note over S: YOLOv8 nano rulează LOCAL pe server<br/>Fără apel extern, fără cost, offline

    S->>S: _model(img, verbose=False)\nYOLOv8 nano detectează toate obiectele din COCO

    S->>S: Filtru clase COCO sensibile:\n62 (TV) → "screen"\n63 (laptop) → "screen"\n67 (cell phone) → "screen"\n73 (carte/book) → "document"\nALTE clase: ignorate

    S->>S: Filtru confidence ≥ 0.40\n(sub prag = obiect nesigur, ignorat)

    S->>S: apply_blur() — același cod ca Gemini

    S-->>A: HTTP 200 JPEG blurat
    A->>U: Afișează rezultat
```

#### De ce kernel 51×51?

```
Kernel mic (5×5):  blur slab, text poate fi citit
Kernel mediu (21×21): blur mediu, text greu de citit
Kernel 51×51:     blur puternic — conținut complet nerecognoscibil ✓
Kernel 101×101:   overkill, procesare mai lentă fără beneficiu real
```

Sigma = 0 înseamnă că OpenCV calculează sigma automat din dimensiunea kernel-ului: `σ = 0.3 × ((51-1)/2 - 1) + 0.8 ≈ 7.7`

---

### 23. Cum funcționează mail — detaliu complet

```mermaid
sequenceDiagram
    actor U as Utilizator
    participant A as Android
    participant S as Server FastAPI
    participant SMTP as SMTP Server (Gmail)

    U->>A: Selectează fotografii din galerie
    U->>A: Introduce: to_email, subject, mesaj

    A->>A: Pentru fiecare fotografie:\n  bytes = ContentResolver.openInputStream(uri)\n  RequestBody.create(bytes, MediaType("image/jpeg"))\n  MultipartBody.Part.createFormData("files", filename, body)

    A->>S: POST /mail/send\nAuthorization: Bearer JWT\nContent-Type: multipart/form-data\n\nto_email=destinatar@example.com\nsubject=Fotografiile tale\nmessage=Salut...\nfiles[0]=<JPEG bytes>\nfiles[1]=<JPEG bytes>

    S->>S: JWT decode → current_user.email

    Note over S: Credențiale SMTP din variabile de mediu:<br/>SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD<br/>NU sunt în cod, NU sunt în request

    S->>S: msg = MIMEMultipart()\nmsg["From"] = SMTP_USER\nmsg["To"] = to_email\nmsg["Subject"] = subject

    S->>S: msg.attach(MIMEText(\n  message or\n  f"Fotografii trimise de {current_user.email}"\n))

    loop Pentru fiecare fișier din request
        S->>S: data = await f.read()\nimg_part = MIMEImage(data, name=filename)\nimg_part.add_header("Content-Disposition",\n  "attachment", filename=filename)\nmsg.attach(img_part)
    end

    S->>SMTP: smtplib.SMTP(host, port=587)
    S->>SMTP: server.ehlo()
    S->>SMTP: server.starttls() ← negociază TLS
    S->>SMTP: server.login(SMTP_USER, SMTP_PASSWORD)
    S->>SMTP: server.sendmail(SMTP_USER, to_email, msg.as_string())

    SMTP-->>U: Email livrat cu fotografiile atașate

    S-->>A: HTTP 200\n{"sent": true, "to": "dest@...", "photos_attached": 2}
    A->>U: Toast: "Email trimis — 2 fotografii atașate"
```

#### Structura email-ului generat

```
From: SMTP_USER (configurat în env)
To: adresa introdusă de utilizator
Subject: subiectul introdus de utilizator

Body (text/plain):
  Fotografii trimise de utilizator@email.com
  [sau mesajul personalizat al utilizatorului]

Attachment 1: photo_1.jpg (image/jpeg)
Attachment 2: photo_2.jpg (image/jpeg)
...
```

#### De ce STARTTLS și nu SSL direct?

```
Port 465 / SSL direct  → conexiune criptată de la bun început
Port 587 / STARTTLS    → conexiune plain → upgrade la TLS mid-session

Am ales STARTTLS (587) pentru că:
• Este standardul modern recomandat de RFC 8314
• Gmail, Outlook, și aproape toți providerii îl suportă
• smtplib.SMTP + starttls() este mai simplu de implementat cross-provider
  decât smtplib.SMTP_SSL care necesită context SSL explicit
```

---

*Diagrame generate conform stării codului din ramura `refactorizare` la data ultimului commit.*
