package com.photomatch;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.gson.Gson;
import com.photomatch.api.BatchResponse;
import com.photomatch.api.BatchResult;
import com.photomatch.api.SearchAndCorrectRequest;
import com.photomatch.api.SearchAndCorrectResponse;
import com.photomatch.base.BaseServerActivity;
import com.photomatch.ml.CLIPEncoder;
import com.photomatch.ml.DefectDetector;
import com.photomatch.ml.HybridVectorBuilder;
import com.photomatch.ml.ImageCorrector;
import com.photomatch.ml.LutCache;
import com.photomatch.ui.ThumbnailAdapter;
import com.photomatch.util.ImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;

import retrofit2.Response;

public class BatchActivity extends BaseServerActivity {

    private static final int MAX_PHOTOS = 100;
    static final String EXTRA_CACHE_PATH = "cache_path";

    private List<Uri>        selectedUris = new ArrayList<>();
    private Button           btnPick;
    private Button           btnProcess;
    private TextView         tvCount;
    private TextView         tvProgress;
    private TextView         tvError;
    private ThumbnailAdapter thumbnailAdapter;

    private final ActivityResultLauncher<String> pickLauncher =
        registerForActivityResult(new ActivityResultContracts.GetMultipleContents(), uris -> {
            if (uris != null && !uris.isEmpty()) {
                selectedUris = new ArrayList<>(uris.subList(0, Math.min(uris.size(), MAX_PHOTOS)));
                updateSelectionUI();
            }
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_batch);

        btnPick      = findViewById(R.id.btnPick);
        btnProcess   = primaryButton = findViewById(R.id.btnProcess);
        tvCount      = findViewById(R.id.tvCount);
        tvProgress   = findViewById(R.id.tvProgress);
        tvError      = findViewById(R.id.tvError);
        progressBar  = findViewById(R.id.progressBar);

        RecyclerView rvThumbnails = findViewById(R.id.rvThumbnails);
        thumbnailAdapter = new ThumbnailAdapter(getContentResolver());
        rvThumbnails.setLayoutManager(new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false));
        rvThumbnails.setAdapter(thumbnailAdapter);

        btnPick.setOnClickListener(v -> pickLauncher.launch("image/*"));
        btnProcess.setOnClickListener(v -> startBatchProcessing());
        updateProcessButton();
    }

    private void updateSelectionUI() {
        int n = selectedUris.size();
        tvCount.setText(n + " photo" + (n == 1 ? "" : "s") + " selected");
        thumbnailAdapter.setUris(selectedUris);
        updateProcessButton();
    }

    private void updateProcessButton() {
        int n = selectedUris.size();
        if (n == 0) {
            btnProcess.setText("PROCESS 0 PHOTOS");
            btnProcess.setEnabled(false);
        } else {
            btnProcess.setText("PROCESS " + n + " PHOTO" + (n == 1 ? "" : "S"));
            btnProcess.setEnabled(true);
        }
    }

    private void setUiProcessing(boolean processing) {
        btnPick.setEnabled(!processing);
        setProcessing(processing);
        tvProgress.setVisibility(processing ? View.VISIBLE : View.GONE);
        tvError.setVisibility(View.GONE);
    }

    private void startBatchProcessing() {
        if (selectedUris.isEmpty()) return;
        setUiProcessing(true);
        progressBar.setMax(selectedUris.size());
        progressBar.setIndeterminate(false);

        executor.execute(() -> {
            try {
                runOnUiThread(() -> tvProgress.setText("Loading CLIP model..."));
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

                DefectDetector defectDetector = new DefectDetector(this);
                List<BatchResult> results   = new ArrayList<>();
                List<Boolean>     blurFlags = new ArrayList<>();
                int failed = 0;

                for (int i = 0; i < selectedUris.size(); i++) {
                    final int idx   = i;
                    final int total = selectedUris.size();
                    runOnUiThread(() -> {
                        tvProgress.setText("Processing " + (idx + 1) + "/" + total + "...");
                        progressBar.setProgress(idx + 1);
                    });

                    Uri uri = selectedUris.get(i);
                    try {
                        Bitmap bmp = ImageUtils.decodeBitmap(getContentResolver(), uri, 1200);
                        blurFlags.add(BlurDetector.check(bmp).isBlurry);

                        float[] clip   = clipEncoder.encode(bmp);
                        float[] defect = defectDetector.detect(bmp);
                        float[] hybrid = HybridVectorBuilder.build(clip, defect);

                        Response<SearchAndCorrectResponse> searchResp =
                            api().searchAndCorrect(new SearchAndCorrectRequest(hybrid), 0.3f, false).execute();
                        if (!searchResp.isSuccessful() || searchResp.body() == null) {
                            bmp.recycle(); failed++; continue;
                        }
                        SearchAndCorrectResponse body = searchResp.body();

                        float[] lut = LutCache.get(this, body.retrieved);
                        Bitmap corrected = ImageCorrector.correct(bmp, lut);
                        bmp.recycle();

                        File outFile = saveCorrectedToFile(corrected, i);
                        corrected.recycle();

                        BatchResult result = new BatchResult();
                        result.index               = i;
                        result.retrieved           = body.retrieved;
                        result.similarity          = body.similarity;
                        result.correctedPath       = outFile != null ? outFile.getAbsolutePath() : null;
                        result.matchAestheticScore = body.matchAestheticScore;
                        results.add(result);

                    } catch (Exception e) {
                        failed++;
                        if (blurFlags.size() <= i) blurFlags.add(false);
                    }
                }

                clipEncoder.close();
                defectDetector.close();

                BatchResponse batchResponse = new BatchResponse();
                batchResponse.results   = results;
                batchResponse.processed = results.size();
                batchResponse.failed    = failed;

                BatchCache cache    = new BatchCache();
                cache.response      = batchResponse;
                cache.originalUris  = urisToStrings(selectedUris);
                cache.blurFlags     = blurFlags;
                String json = new Gson().toJson(cache);

                File outFile = new File(getExternalFilesDir(null),
                    "batch_" + System.currentTimeMillis() + ".json");
                try (FileWriter fw = new FileWriter(outFile)) { fw.write(json); }

                Intent intent = new Intent(this, BatchResultsActivity.class);
                intent.putExtra(EXTRA_CACHE_PATH, outFile.getAbsolutePath());
                runOnUiThread(() -> { startActivity(intent); finish(); });

            } catch (Exception e) {
                runOnUiThread(() -> {
                    setUiProcessing(false);
                    showError("Error: " + e.getMessage());
                });
            }
        });
    }

    private File saveCorrectedToFile(Bitmap bmp, int index) {
        File out = new File(getCacheDir(), "batch_corrected_" + index + ".jpg");
        try (FileOutputStream fos = new FileOutputStream(out)) {
            bmp.compress(Bitmap.CompressFormat.JPEG, 85, fos);
            return out;
        } catch (IOException e) {
            return null;
        }
    }

    private static List<String> urisToStrings(List<Uri> uris) {
        List<String> out = new ArrayList<>(uris.size());
        for (Uri u : uris) out.add(u.toString());
        return out;
    }

    public static class BatchCache {
        public BatchResponse response;
        public List<String>  originalUris;
        public List<Boolean> blurFlags;
    }
}
