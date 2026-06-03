package com.photomatch;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.photomatch.api.ApiClient;
import com.photomatch.api.ProcessResponse;
import com.photomatch.api.SearchAndCorrectRequest;
import com.photomatch.api.SearchAndCorrectResponse;
import com.photomatch.api.StyleSearchRequest;
import com.photomatch.api.StyleSearchResponse;
import com.photomatch.base.BaseServerActivity;
import com.photomatch.ml.CLIPEncoder;
import com.photomatch.ml.DefectDetector;
import com.photomatch.ml.HybridVectorBuilder;
import com.photomatch.ml.ImageCorrector;
import com.photomatch.ml.LutCache;
import com.photomatch.util.ImageUtils;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;

import retrofit2.Response;

public class ProcessingActivity extends BaseServerActivity {

    private static final String TAG = "PM_Latency";

    private static final String[] LOG_LINES = {
        "extracting visual semantics_",
        "measuring technical defects_",
        "searching 3499 reference photographs_",
        "applying expert colour grade_",
    };

    private String imagePath;
    private boolean useStyle;
    private String  sessionId;

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

        tvLog       = findViewById(R.id.tvLog);
        tvError     = findViewById(R.id.tvError);
        btnRetry    = findViewById(R.id.btnRetry);
        bannerBlur  = findViewById(R.id.bannerBlur);
        tvBlurScore = findViewById(R.id.tvBlurScore);

        ImageButton btnDismissBlur = findViewById(R.id.btnDismissBlur);
        btnDismissBlur.setOnClickListener(v -> bannerBlur.setVisibility(View.GONE));

        ImageView ivPreview = findViewById(R.id.ivPreview);
        Glide.with(this).load(new File(imagePath))
            .diskCacheStrategy(DiskCacheStrategy.NONE).skipMemoryCache(true)
            .fitCenter().into(ivPreview);

        btnRetry.setOnClickListener(v -> {
            Intent intent = new Intent(this, ProcessingActivity.class);
            intent.putExtra(MainActivity.EXTRA_IMAGE_PATH, imagePath);
            startActivity(intent);
            finish();
        });

        startPipeline();
    }

    private void startPipeline() {
        executor.execute(() -> {
            try {
                final long t0 = SystemClock.elapsedRealtime();

                Bitmap preview = BitmapFactory.decodeFile(imagePath);
                if (preview != null) {
                    BlurDetector.BlurResult blur = BlurDetector.check(preview);
                    preview.recycle();
                    if (blur.isBlurry) runOnUiThread(() -> showBlurBanner(blur.score));
                }

                appendLog(LOG_LINES[0]);

                Bitmap bmp = ImageUtils.decodeBitmap(imagePath, 512);

                CLIPEncoder clipEncoder = new CLIPEncoder(this);
                CountDownLatch latch = new CountDownLatch(1);
                Exception[] loadErr = {null};
                clipEncoder.loadAsync(Executors.newSingleThreadExecutor(),
                    new CLIPEncoder.LoadCallback() {
                        @Override public void onLoaded()           { latch.countDown(); }
                        @Override public void onError(Exception e) { loadErr[0] = e; latch.countDown(); }
                    });
                latch.await();
                if (loadErr[0] != null) throw loadErr[0];

                long tClip = SystemClock.elapsedRealtime();
                float[] clipVec = clipEncoder.encode(bmp);
                Log.i(TAG, "CLIP encode: " + (SystemClock.elapsedRealtime() - tClip) + " ms");
                clipEncoder.close();

                appendLog(LOG_LINES[1]);

                DefectDetector defectDetector = new DefectDetector(this);
                long tDefect = SystemClock.elapsedRealtime();
                float[] defectVec = defectDetector.detect(bmp);
                Log.i(TAG, "Defect detect: " + (SystemClock.elapsedRealtime() - tDefect) + " ms");
                defectDetector.close();
                bmp.recycle();

                float[] hybrid = HybridVectorBuilder.build(clipVec, defectVec);

                appendLog(LOG_LINES[2]);

                String basename;
                float similarity;
                float matchAes;
                long tNet = SystemClock.elapsedRealtime();
                if (useStyle && sessionId != null) {
                    Response<StyleSearchResponse> resp =
                        api().styleSearch(new StyleSearchRequest(hybrid, sessionId)).execute();
                    if (!resp.isSuccessful() || resp.body() == null)
                        throw new IOException("Style search failed: HTTP " + resp.code());
                    basename   = resp.body().retrieved;
                    similarity = resp.body().similarity;
                    matchAes   = 0f;
                } else {
                    Response<SearchAndCorrectResponse> resp =
                        api().searchAndCorrect(new SearchAndCorrectRequest(hybrid), 0.3f, false).execute();
                    if (!resp.isSuccessful() || resp.body() == null)
                        throw new IOException("Search failed: HTTP " + resp.code());
                    basename   = resp.body().retrieved;
                    similarity = resp.body().similarity;
                    matchAes   = resp.body().matchAestheticScore;
                }
                Log.i(TAG, "Network RTT: " + (SystemClock.elapsedRealtime() - tNet) + " ms");

                appendLog(LOG_LINES[3]);

                Bitmap fullBmp = BitmapFactory.decodeFile(imagePath);
                if (fullBmp == null) throw new IOException("Could not decode full image");

                float[] lut = LutCache.get(this, basename);
                Bitmap corrected = ImageCorrector.correct(fullBmp, lut);
                String note = (lut != null) ? "" : "LUT not available — CLAHE correction only";

                String[] defectNames = {"blur", "noise", "overexposure", "underexposure", "compression"};
                Map<String, Float> defects = new HashMap<>();
                for (int i = 0; i < defectNames.length && i < defectVec.length; i++)
                    defects.put(defectNames[i], defectVec[i]);

                ProcessResponse response = new ProcessResponse();
                response.originalB64         = ApiClient.bitmapToBase64(fullBmp);
                response.correctedB64        = ApiClient.bitmapToBase64(corrected);
                response.finalB64            = response.correctedB64;
                response.defects             = defects;
                response.retrieved           = basename;
                response.similarity          = similarity;
                response.rawB64              = "";
                response.editedB64           = "";
                response.note                = note;
                response.matchAestheticScore = matchAes;

                fullBmp.recycle();
                corrected.recycle();
                Log.i(TAG, "Pipeline total: " + (SystemClock.elapsedRealtime() - t0) + " ms");

                ResponseCache.current = response;
                runOnUiThread(() -> { startActivity(new Intent(this, ResultsActivity.class)); finish(); });

            } catch (Exception e) {
                runOnUiThread(() -> showPipelineError("Error: " + e.getMessage()));
            }
        });
    }

    private void appendLog(String line) {
        runOnUiThread(() -> tvLog.append(line + "\n"));
    }

    private void showBlurBanner(float score) {
        tvBlurScore.setText(String.format(java.util.Locale.US, "Sharpness: %.1f", score));
        bannerBlur.setVisibility(View.VISIBLE);
    }

    private void showPipelineError(String message) {
        tvError.setText(message);
        tvError.setVisibility(View.VISIBLE);
        btnRetry.setVisibility(View.VISIBLE);
    }
}
