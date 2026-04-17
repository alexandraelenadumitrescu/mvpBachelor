package com.photomatch;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.photomatch.api.ApiClient;
import com.photomatch.api.ProcessResponse;
import com.photomatch.api.SearchAndCorrectRequest;
import com.photomatch.api.SearchAndCorrectResponse;
import com.photomatch.api.StyleSearchRequest;
import com.photomatch.api.StyleSearchResponse;
import com.photomatch.ml.CLIPEncoder;
import com.photomatch.ml.DefectDetector;
import com.photomatch.ml.HybridVectorBuilder;
import com.photomatch.ml.ImageCorrector;
import com.photomatch.ml.LutCache;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import retrofit2.Response;

/**
 * V2 — images never leave the device.
 * Pipeline: on-device CLIP + defect → /search_and_correct (vector only)
 * → download/cache LUT → ImageCorrector on-device → ResultsActivity
 */
public class ProcessingActivity extends AppCompatActivity {

    private static final String TAG = "PM_Latency";

    private static final String[] LOG_LINES = {
        "extracting visual semantics_",
        "measuring technical defects_",
        "searching 3499 reference photographs_",
        "applying expert colour grade_",
    };

    private String  imagePath;
    private boolean useStyle;
    private String  sessionId;
    private ExecutorService executor;

    private TextView     tvLog;
    private TextView     tvError;
    private Button       btnRetry;
    private LinearLayout bannerBlur;
    private TextView     tvBlurScore;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_processing);

        imagePath = getIntent().getStringExtra(MainActivity.EXTRA_IMAGE_PATH);
        if (imagePath == null) { finish(); return; }
        useStyle  = getIntent().getBooleanExtra("use_style", false);
        sessionId = getIntent().getStringExtra("session_id");

        tvLog      = findViewById(R.id.tvLog);
        tvError    = findViewById(R.id.tvError);
        btnRetry   = findViewById(R.id.btnRetry);
        bannerBlur = findViewById(R.id.bannerBlur);
        tvBlurScore = findViewById(R.id.tvBlurScore);
        ImageButton btnDismissBlur = findViewById(R.id.btnDismissBlur);
        btnDismissBlur.setOnClickListener(v -> bannerBlur.setVisibility(View.GONE));

        ImageView ivPreview = findViewById(R.id.ivPreview);
        Glide.with(this)
            .load(new File(imagePath))
            .diskCacheStrategy(DiskCacheStrategy.NONE)
            .skipMemoryCache(true)
            .fitCenter()
            .into(ivPreview);

        btnRetry.setOnClickListener(v -> {
            Intent intent = new Intent(this, ProcessingActivity.class);
            intent.putExtra(MainActivity.EXTRA_IMAGE_PATH, imagePath);
            startActivity(intent);
            finish();
        });

        startPipeline();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor != null) executor.shutdownNow();
    }

    private void startPipeline() {
        executor = Executors.newSingleThreadExecutor();
        executor.execute(() -> {
            try {
                final long tPipelineStart = SystemClock.elapsedRealtime();

                // Blur check
                Bitmap preview = BitmapFactory.decodeFile(imagePath);
                if (preview != null) {
                    BlurDetector.BlurResult blur = BlurDetector.check(preview);
                    preview.recycle();
                    if (blur.isBlurry) runOnUiThread(() -> showBlurBanner(blur.score));
                }

                appendLog(LOG_LINES[0]); // extracting visual semantics_

                // Load image for vector computation (512px side for speed)
                Bitmap bmp = decodeBitmap(imagePath, 512);
                if (bmp == null) throw new IOException("Could not decode image");

                // On-device CLIP encoding
                CLIPEncoder clipEncoder = new CLIPEncoder(this);
                CountDownLatch latch = new CountDownLatch(1);
                Exception[] loadErr = {null};
                clipEncoder.loadAsync(Executors.newSingleThreadExecutor(),
                    new CLIPEncoder.LoadCallback() {
                        @Override public void onLoaded()          { latch.countDown(); }
                        @Override public void onError(Exception e) { loadErr[0] = e; latch.countDown(); }
                    });
                latch.await();
                if (loadErr[0] != null) throw loadErr[0];

                long tClip = SystemClock.elapsedRealtime();
                float[] clipVec = clipEncoder.encode(bmp);
                long clipMs = SystemClock.elapsedRealtime() - tClip;
                clipEncoder.close();
                Log.i(TAG, "CLIP encode: " + clipMs + " ms");

                appendLog(LOG_LINES[1]); // measuring technical defects_

                // On-device defect detection
                DefectDetector defectDetector = new DefectDetector(this);
                long tDefect = SystemClock.elapsedRealtime();
                float[] defectVec = defectDetector.detect(bmp);
                long defectMs = SystemClock.elapsedRealtime() - tDefect;
                defectDetector.close();
                bmp.recycle();
                Log.i(TAG, "Defect detect: " + defectMs + " ms");

                // Build 517-dim hybrid vector
                long tHybrid = SystemClock.elapsedRealtime();
                float[] hybrid = HybridVectorBuilder.build(clipVec, defectVec);
                Log.i(TAG, "Hybrid build: " + (SystemClock.elapsedRealtime() - tHybrid) + " ms");

                appendLog(LOG_LINES[2]); // searching 3499 reference photographs_

                // Server call — vector only, no image data
                String basename;
                float similarity;
                float matchAes;
                long tNetwork = SystemClock.elapsedRealtime();
                if (useStyle && sessionId != null) {
                    StyleSearchRequest req = new StyleSearchRequest(hybrid, sessionId);
                    Response<StyleSearchResponse> resp =
                        ApiClient.getInstance().getService().styleSearch(req).execute();
                    if (!resp.isSuccessful() || resp.body() == null)
                        throw new IOException("Style search failed: HTTP " + resp.code());
                    basename   = resp.body().retrieved;
                    similarity = resp.body().similarity;
                    matchAes   = 0f;
                } else {
                    SearchAndCorrectRequest req = new SearchAndCorrectRequest(hybrid);
                    Response<SearchAndCorrectResponse> resp =
                        ApiClient.getInstance().getService()
                            .searchAndCorrect(req, 0.3f, false).execute();
                    if (!resp.isSuccessful() || resp.body() == null)
                        throw new IOException("Search failed: HTTP " + resp.code());
                    basename   = resp.body().retrieved;
                    similarity = resp.body().similarity;
                    matchAes   = resp.body().matchAestheticScore;
                }
                Log.i(TAG, "Network /search_and_correct RTT: " + (SystemClock.elapsedRealtime() - tNetwork) + " ms");

                appendLog(LOG_LINES[3]); // applying expert colour grade_

                // Load full-res image for correction
                Bitmap fullBmp = BitmapFactory.decodeFile(imagePath);
                if (fullBmp == null) throw new IOException("Could not decode full image");

                // Download / cache LUT, apply on-device
                long tLut = SystemClock.elapsedRealtime();
                float[] lut = LutCache.get(this, basename);
                Log.i(TAG, "LUT download/cache: " + (SystemClock.elapsedRealtime() - tLut) + " ms");

                long tCorrect = SystemClock.elapsedRealtime();
                Bitmap corrected = ImageCorrector.correct(fullBmp, lut);
                Log.i(TAG, "ImageCorrector.correct: " + (SystemClock.elapsedRealtime() - tCorrect) + " ms");
                String note = (lut != null) ? "" : "LUT not available — CLAHE correction only";

                // Build defect map for ResultsActivity
                String[] defectNames = {"blur", "noise", "overexposure", "underexposure", "compression"};
                Map<String, Float> defects = new HashMap<>();
                for (int i = 0; i < defectNames.length && i < defectVec.length; i++)
                    defects.put(defectNames[i], defectVec[i]);

                // Encode images to base64
                String originalB64  = bitmapToBase64(fullBmp);
                String correctedB64 = bitmapToBase64(corrected);
                fullBmp.recycle();
                corrected.recycle();

                // Populate response and navigate
                ProcessResponse response = new ProcessResponse();
                response.originalB64          = originalB64;
                response.correctedB64         = correctedB64;
                response.finalB64             = correctedB64;
                response.defects              = defects;
                response.retrieved            = basename;
                response.similarity           = similarity;
                response.rawB64               = "";
                response.editedB64            = "";
                response.note                 = note;
                response.matchAestheticScore  = matchAes;

                Log.i(TAG, "Pipeline total: " + (SystemClock.elapsedRealtime() - tPipelineStart) + " ms");

                ResponseCache.current = response;
                runOnUiThread(() -> {
                    startActivity(new Intent(this, ResultsActivity.class));
                    finish();
                });

            } catch (Exception e) {
                runOnUiThread(() -> showError("Error: " + e.getMessage()));
            }
        });
    }

    // ── helpers ────────────────────────────────────────────────────────────────

    private Bitmap decodeBitmap(String path, int maxSide) {
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(path, opts);
        opts.inSampleSize = MainActivity.computeSampleSize(opts.outWidth, opts.outHeight, maxSide);
        opts.inJustDecodeBounds = false;
        return BitmapFactory.decodeFile(path, opts);
    }

    private String bitmapToBase64(Bitmap bmp) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 90, out);
        return Base64.encodeToString(out.toByteArray(), Base64.DEFAULT);
    }

    private void appendLog(String line) {
        runOnUiThread(() -> tvLog.append(line + "\n"));
    }

    private void showBlurBanner(float score) {
        tvBlurScore.setText(String.format(java.util.Locale.US, "Sharpness: %.1f", score));
        bannerBlur.setVisibility(View.VISIBLE);
    }

    private void showError(String message) {
        tvError.setText(message);
        tvError.setVisibility(View.VISIBLE);
        btnRetry.setVisibility(View.VISIBLE);
    }
}
