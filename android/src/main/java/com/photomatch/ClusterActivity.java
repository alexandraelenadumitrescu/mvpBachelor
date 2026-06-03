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
import com.photomatch.api.ClusterRequest;
import com.photomatch.api.ClusterResponse;
import com.photomatch.base.BaseServerActivity;
import com.photomatch.ml.CLIPEncoder;
import com.photomatch.ml.DefectDetector;
import com.photomatch.ml.HybridVectorBuilder;
import com.photomatch.ui.ThumbnailAdapter;
import com.photomatch.util.ImageUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import retrofit2.Response;

public class ClusterActivity extends BaseServerActivity {

    private static final int MAX_PHOTOS = 200;
    static final String EXTRA_CACHE_PATH = "cluster_cache_path";

    private List<Uri>        selectedUris = new ArrayList<>();
    private CLIPEncoder      clipEncoder;
    private DefectDetector   defectDetector;
    private boolean          modelReady   = false;

    private Button           btnPick;
    private Button           btnAnalyze;
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
        setContentView(R.layout.activity_cluster);

        btnPick    = findViewById(R.id.btnPick);
        btnAnalyze = primaryButton = findViewById(R.id.btnAnalyze);
        tvCount    = findViewById(R.id.tvCount);
        tvProgress = findViewById(R.id.tvProgress);
        tvError    = findViewById(R.id.tvError);
        progressBar = findViewById(R.id.progressBar);

        RecyclerView rvThumbnails = findViewById(R.id.rvThumbnails);
        thumbnailAdapter = new ThumbnailAdapter(getContentResolver());
        rvThumbnails.setLayoutManager(new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false));
        rvThumbnails.setAdapter(thumbnailAdapter);

        btnPick.setOnClickListener(v -> pickLauncher.launch("image/*"));
        btnAnalyze.setOnClickListener(v -> startAnalysis());

        defectDetector = new DefectDetector(this);

        try {
            clipEncoder = new CLIPEncoder(this);
            btnAnalyze.setText("Loading model...");
            clipEncoder.loadAsync(executor, new CLIPEncoder.LoadCallback() {
                @Override public void onLoaded() {
                    modelReady = true;
                    runOnUiThread(() -> updateAnalyzeButton());
                }
                @Override public void onError(Exception e) {
                    runOnUiThread(() -> showError("CLIP model error: " + e.getMessage()));
                }
            });
        } catch (IllegalStateException e) {
            showError("clip_model.gguf not found on device");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy(); // shuts down executor
        if (clipEncoder    != null) clipEncoder.close();
        if (defectDetector != null) defectDetector.close();
    }

    private void updateSelectionUI() {
        int n = selectedUris.size();
        tvCount.setText(n + " photo" + (n == 1 ? "" : "s") + " selected");
        thumbnailAdapter.setUris(selectedUris);
        updateAnalyzeButton();
    }

    private void updateAnalyzeButton() {
        int n = selectedUris.size();
        if (!modelReady) {
            btnAnalyze.setText("Loading model...");
            btnAnalyze.setEnabled(false);
        } else if (n == 0) {
            btnAnalyze.setText("ANALYZE 0 PHOTOS");
            btnAnalyze.setEnabled(false);
        } else {
            btnAnalyze.setText("ANALYZE " + n + " PHOTO" + (n == 1 ? "" : "S"));
            btnAnalyze.setEnabled(true);
        }
    }

    private void setUiProcessing(boolean processing) {
        btnPick.setEnabled(!processing);
        setProcessing(processing);
        tvProgress.setVisibility(processing ? View.VISIBLE : View.GONE);
        tvError.setVisibility(View.GONE);
    }

    private void startAnalysis() {
        if (selectedUris.isEmpty() || !modelReady) return;
        setUiProcessing(true);
        progressBar.setMax(selectedUris.size());

        executor.execute(() -> {
            try {
                List<List<Float>> vectors = new ArrayList<>();
                for (int i = 0; i < selectedUris.size(); i++) {
                    final int idx = i;
                    runOnUiThread(() -> {
                        tvProgress.setText("Analyzing " + (idx + 1) + "/" + selectedUris.size() + "...");
                        progressBar.setProgress(idx + 1);
                    });

                    Bitmap bmp     = ImageUtils.decodeBitmap(getContentResolver(), selectedUris.get(i), 512);
                    float[] clip   = clipEncoder.encode(bmp);
                    float[] defect = defectDetector.detect(bmp);
                    float[] hybrid = HybridVectorBuilder.build(clip, defect);
                    bmp.recycle();
                    vectors.add(toFloatList(hybrid));
                }

                runOnUiThread(() -> {
                    tvProgress.setText("Sending to server...");
                    progressBar.setIndeterminate(true);
                });

                ClusterRequest req = new ClusterRequest();
                req.vectors   = vectors;
                req.nClusters = null;

                Response<ClusterResponse> resp = api().cluster(req).execute();
                if (!resp.isSuccessful() || resp.body() == null) {
                    String err = resp.errorBody() != null ? resp.errorBody().string() : "";
                    throw new IOException("Server error HTTP " + resp.code() + " " + err);
                }

                ClusterCache cache    = new ClusterCache();
                cache.response        = resp.body();
                cache.originalUris    = urisToStrings(selectedUris);
                String json = new Gson().toJson(cache);

                File outFile = new File(getExternalFilesDir(null),
                    "cluster_" + System.currentTimeMillis() + ".json");
                try (FileWriter fw = new FileWriter(outFile)) { fw.write(json); }

                Intent intent = new Intent(this, ClusterResultsActivity.class);
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

    private static List<Float> toFloatList(float[] arr) {
        List<Float> list = new ArrayList<>(arr.length);
        for (float v : arr) list.add(v);
        return list;
    }

    private static List<String> urisToStrings(List<Uri> uris) {
        List<String> out = new ArrayList<>(uris.size());
        for (Uri u : uris) out.add(u.toString());
        return out;
    }

    public static class ClusterCache {
        public ClusterResponse response;
        public List<String>    originalUris;
    }
}
