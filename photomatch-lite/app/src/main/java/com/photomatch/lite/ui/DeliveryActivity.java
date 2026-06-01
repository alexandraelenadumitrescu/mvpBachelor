package com.photomatch.lite.ui;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import java.util.List;
import androidx.appcompat.app.AppCompatActivity;
import com.google.android.material.textfield.TextInputEditText;
import com.photomatch.lite.R;
import com.photomatch.lite.api.ApiClient;
import com.photomatch.lite.api.DeliveryDetail;
import com.photomatch.lite.api.DeliveryResponse;
import com.photomatch.lite.api.MatchRequest;
import com.photomatch.lite.api.MatchResponse;
import com.photomatch.lite.api.PhotoEmbeddings;
import com.photomatch.lite.api.PhotoMatch;
import com.photomatch.lite.face.FaceClusterer;
import com.photomatch.lite.face.FaceDetectorHelper;
import com.photomatch.lite.face.FaceEmbedder;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class DeliveryActivity extends AppCompatActivity {

    private TextInputEditText etUrl;
    private TextView          tvCount;
    private TextView          tvProgress;
    private TextView          tvStatus;
    private ProgressBar       progressBar;
    private Button            btnPick;
    private Button            btnClear;
    private Button            btnSend;

    private List<Uri>          selectedUris = new ArrayList<>();
    private ExecutorService    executor;
    private FaceDetectorHelper faceDetector;
    private FaceEmbedder       faceEmbedder;

    private final ActivityResultLauncher<String> pickPhotos =
        registerForActivityResult(new ActivityResultContracts.GetMultipleContents(), uris -> {
            selectedUris.addAll(uris);
            tvCount.setText(selectedUris.size() + " fotografii selectate");
        });

    private final ActivityResultLauncher<Uri> pickFolder =
        registerForActivityResult(new ActivityResultContracts.OpenDocumentTree(), uri -> {
            if (uri == null) return;
            List<Uri> images = FolderPickerHelper.listImages(this, uri);
            selectedUris.addAll(images);
            tvCount.setText(selectedUris.size() + " fotografii selectate");
            Toast.makeText(this, images.size() + " fotografii adăugate din folder",
                Toast.LENGTH_SHORT).show();
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_delivery);

        etUrl       = findViewById(R.id.etUrl);
        tvCount     = findViewById(R.id.tvCount);
        tvProgress  = findViewById(R.id.tvProgress);
        tvStatus    = findViewById(R.id.tvStatus);
        progressBar = findViewById(R.id.progressBar);
        btnPick     = findViewById(R.id.btnPick);
        btnClear    = findViewById(R.id.btnClear);
        btnSend     = findViewById(R.id.btnSend);

        etUrl.setText("https://alexandraelenadumitrescu.github.io/nexus.html");

        executor     = Executors.newSingleThreadExecutor();
        faceDetector = new FaceDetectorHelper();
        try {
            faceEmbedder = new FaceEmbedder(this);
        } catch (RuntimeException e) {
            Toast.makeText(this, "facenet.tflite lipsă", Toast.LENGTH_LONG).show();
        }

        btnPick.setOnClickListener(v -> pickPhotos.launch("image/*"));
        findViewById(R.id.btnPickFolder).setOnClickListener(v -> pickFolder.launch(null));
        btnClear.setOnClickListener(v -> {
            selectedUris.clear();
            tvCount.setText("0 fotografii selectate");
        });
        btnSend.setOnClickListener(v -> startDelivery());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor     != null) executor.shutdownNow();
        if (faceDetector != null) faceDetector.close();
        if (faceEmbedder != null) faceEmbedder.close();
    }

    private void startDelivery() {
        String url = etUrl.getText() != null ? etUrl.getText().toString().trim() : "";
        if (url.isEmpty()) {
            Toast.makeText(this, "Introduceți URL-ul angajaților", Toast.LENGTH_SHORT).show();
            return;
        }
        if (selectedUris.isEmpty()) {
            Toast.makeText(this, "Selectează fotografii mai întâi", Toast.LENGTH_SHORT).show();
            return;
        }
        if (faceEmbedder == null) {
            Toast.makeText(this, "Modelul nu a putut fi încărcat", Toast.LENGTH_SHORT).show();
            return;
        }

        setUiLoading(true);
        tvStatus.setText("");

        executor.execute(() -> phase1_embedLocally(url));
    }

    // ── Faza 1: procesare locală ─────────────────────────────────────────────

    private void phase1_embedLocally(String employeesUrl) {
        List<PhotoEmbeddings> photoEmbList = new ArrayList<>();
        int total = selectedUris.size();

        for (int i = 0; i < total; i++) {
            final int idx = i;
            runOnUiThread(() -> {
                tvProgress.setText("Faza 1/3 · procesare locală " + (idx + 1) + "/" + total);
                progressBar.setMax(total);
                progressBar.setProgress(idx + 1);
            });

            Uri uri = selectedUris.get(i);
            Bitmap bmp = decodeBitmap(uri, 1024);
            if (bmp == null) continue;

            List<List<Float>> faceEmbs = new ArrayList<>();
            try {
                List<Rect> boxes = faceDetector.detect(bmp);
                for (Rect box : boxes) {
                    Bitmap crop = cropFace(bmp, box);
                    float[] emb = faceEmbedder.embed(crop);
                    crop.recycle();
                    List<Float> embList = new ArrayList<>(emb.length);
                    for (float f : emb) embList.add(f);
                    faceEmbs.add(embList);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            bmp.recycle();

            if (!faceEmbs.isEmpty()) {
                photoEmbList.add(new PhotoEmbeddings(i, faceEmbs));
            }
        }

        if (photoEmbList.isEmpty()) {
            runOnUiThread(() -> {
                setUiLoading(false);
                tvStatus.setText("Nu s-au detectat fețe în fotografiile selectate.");
            });
            return;
        }

        runOnUiThread(() -> tvProgress.setText("Faza 2/3 · matching pe server..."));
        phase2_matchOnServer(employeesUrl, photoEmbList);
    }

    // ── Faza 2: matching pe server ───────────────────────────────────────────

    private void phase2_matchOnServer(String employeesUrl, List<PhotoEmbeddings> photoEmbList) {
        MatchRequest req = new MatchRequest(employeesUrl, photoEmbList);
        ApiClient.service().matchEmbeddings(req).enqueue(new Callback<MatchResponse>() {
            @Override
            public void onResponse(Call<MatchResponse> call, Response<MatchResponse> resp) {
                if (!resp.isSuccessful() || resp.body() == null) {
                    runOnUiThread(() -> {
                        setUiLoading(false);
                        tvStatus.setText("Eroare matching: " + resp.code());
                    });
                    return;
                }
                MatchResponse matchResp = resp.body();
                if (matchResp.matches == null || matchResp.matches.isEmpty()) {
                    runOnUiThread(() -> {
                        setUiLoading(false);
                        tvStatus.setText("Nicio persoană recunoscută din "
                            + matchResp.employees_count + " angajați.");
                    });
                    return;
                }
                runOnUiThread(() -> tvProgress.setText(
                    "Faza 3/3 · trimitere " + matchResp.matches.size() + " email-uri..."));
                executor.execute(() -> phase3_sendMatched(matchResp.matches));
            }

            @Override
            public void onFailure(Call<MatchResponse> call, Throwable t) {
                runOnUiThread(() -> {
                    setUiLoading(false);
                    tvStatus.setText("Eroare rețea: " + t.getMessage());
                });
            }
        });
    }

    // ── Faza 3: upload poze matched + trimitere emailuri ────────────────────

    private void phase3_sendMatched(List<PhotoMatch> matches) {
        // group: email → set of photo indices
        Map<String, Set<Integer>> emailToIndices = new HashMap<>();
        for (PhotoMatch m : matches) {
            emailToIndices.computeIfAbsent(m.email, k -> new HashSet<>()).add(m.photo_index);
        }

        int emailsSent = 0, failed = 0;
        final StringBuilder sb = new StringBuilder();
        int current = 0, total = emailToIndices.size();

        for (Map.Entry<String, Set<Integer>> entry : emailToIndices.entrySet()) {
            String      email   = entry.getKey();
            Set<Integer> indices = entry.getValue();
            current++;
            final int cur = current;
            runOnUiThread(() -> tvProgress.setText(
                "Faza 3/3 · email " + cur + "/" + total + " (" + email + ")"));

            // build photo parts for this person only
            List<MultipartBody.Part> parts = new ArrayList<>();
            for (int idx : indices) {
                if (idx >= selectedUris.size()) continue;
                try (InputStream is = getContentResolver().openInputStream(selectedUris.get(idx))) {
                    byte[] bytes = is.readAllBytes();
                    RequestBody body = RequestBody.create(bytes, MediaType.parse("image/jpeg"));
                    parts.add(MultipartBody.Part.createFormData("photos", idx + ".jpg", body));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            if (parts.isEmpty()) continue;

            try {
                RequestBody emailPart = RequestBody.create(email, MediaType.parse("text/plain"));
                Response<DeliveryResponse> resp =
                    ApiClient.service().sendMatched(emailPart, parts).execute();
                if (resp.isSuccessful() && resp.body() != null && resp.body().emails_sent > 0) {
                    emailsSent++;
                    sb.append("\n→ ").append(email)
                      .append(" (").append(indices.size()).append(" poze)");
                } else {
                    failed++;
                }
            } catch (IOException e) {
                failed++;
                e.printStackTrace();
            }
        }

        final int finalSent = emailsSent, finalFailed = failed;
        runOnUiThread(() -> {
            setUiLoading(false);
            tvStatus.setText("Trimise: " + finalSent + " · Eșuate: " + finalFailed + sb);
        });
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private void setUiLoading(boolean on) {
        btnPick.setEnabled(!on);
        btnClear.setEnabled(!on);
        btnSend.setEnabled(!on);
        progressBar.setVisibility(on ? View.VISIBLE : View.GONE);
        tvProgress.setVisibility(on ? View.VISIBLE : View.GONE);
    }

    private Bitmap decodeBitmap(Uri uri, int maxSide) {
        try {
            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            try (InputStream is = getContentResolver().openInputStream(uri)) {
                BitmapFactory.decodeStream(is, null, opts);
            }
            int sample = 1;
            while (opts.outWidth / sample > maxSide || opts.outHeight / sample > maxSide) sample *= 2;
            opts.inSampleSize    = sample;
            opts.inJustDecodeBounds = false;
            try (InputStream is = getContentResolver().openInputStream(uri)) {
                return BitmapFactory.decodeStream(is, null, opts);
            }
        } catch (IOException e) {
            return null;
        }
    }

    private Bitmap cropFace(Bitmap src, Rect box) {
        int pad = (int) (Math.min(box.width(), box.height()) * 0.20f);
        int l = Math.max(0, box.left   - pad);
        int t = Math.max(0, box.top    - pad);
        int r = Math.min(src.getWidth(),  box.right  + pad);
        int b = Math.min(src.getHeight(), box.bottom + pad);
        return Bitmap.createBitmap(src, l, t, r - l, b - t);
    }
}
