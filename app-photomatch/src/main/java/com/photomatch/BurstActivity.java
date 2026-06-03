package com.photomatch;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.gson.Gson;
import com.photomatch.ml.CLIPEncoder;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class BurstActivity extends AppCompatActivity {

    private static final int MAX_PHOTOS = 100;
    static final String EXTRA_CACHE_PATH = "burst_cache_path";

    private List<Uri> selectedUris = new ArrayList<>();
    private CLIPEncoder clipEncoder;
    private boolean     modelReady = false;

    private Button      btnPick;
    private Button      btnAnalyze;
    private TextView    tvCount;
    private TextView    tvProgress;
    private TextView    tvError;
    private ProgressBar progressBar;
    private RecyclerView rvThumbnails;
    private ThumbnailAdapter thumbnailAdapter;

    private ExecutorService executor;

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
        setContentView(R.layout.activity_burst);

        btnPick      = findViewById(R.id.btnPick);
        btnAnalyze   = findViewById(R.id.btnAnalyze);
        tvCount      = findViewById(R.id.tvCount);
        tvProgress   = findViewById(R.id.tvProgress);
        tvError      = findViewById(R.id.tvError);
        progressBar  = findViewById(R.id.progressBar);
        rvThumbnails = findViewById(R.id.rvThumbnails);

        thumbnailAdapter = new ThumbnailAdapter();
        rvThumbnails.setLayoutManager(
            new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false));
        rvThumbnails.setAdapter(thumbnailAdapter);

        btnPick.setOnClickListener(v -> pickLauncher.launch("image/*"));
        btnAnalyze.setOnClickListener(v -> startAnalysis());

        executor = Executors.newSingleThreadExecutor();

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
        super.onDestroy();
        if (executor != null) executor.shutdownNow();
        if (clipEncoder != null) clipEncoder.close();
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

    private void showError(String msg) {
        tvError.setText(msg);
        tvError.setVisibility(View.VISIBLE);
    }

    private void setUiProcessing(boolean processing) {
        btnPick.setEnabled(!processing);
        btnAnalyze.setEnabled(!processing);
        tvProgress.setVisibility(processing ? View.VISIBLE : View.GONE);
        progressBar.setVisibility(processing ? View.VISIBLE : View.GONE);
        tvError.setVisibility(View.GONE);
    }

    private void startAnalysis() {
        if (selectedUris.isEmpty() || !modelReady) return;
        setUiProcessing(true);
        progressBar.setMax(selectedUris.size());
        progressBar.setIndeterminate(false);

        executor.execute(() -> {
            try {
                // Phase 1: encode + score each photo on-device
                List<float[]> clipVectors = new ArrayList<>();
                List<Float>   scores      = new ArrayList<>();

                for (int i = 0; i < selectedUris.size(); i++) {
                    final int idx = i;
                    runOnUiThread(() -> {
                        tvProgress.setText("Analyzing " + (idx + 1) + "/" + selectedUris.size() + "...");
                        progressBar.setProgress(idx + 1);
                    });

                    Bitmap bmp   = decodeBitmap(selectedUris.get(i), 512);
                    float[] clip = clipEncoder.encode(bmp);   // 512-dim, L2-normalised
                    float score  = AestheticScorer.score(bmp);
                    bmp.recycle();

                    clipVectors.add(clip);
                    scores.add(score);
                }

                // Phase 2: cluster locally (no server)
                List<List<Integer>> clusters = BurstClusterer.cluster(clipVectors);

                // Phase 3: persist URI read permissions before this Activity finishes
                for (Uri uri : selectedUris) {
                    try {
                        getContentResolver().takePersistableUriPermission(
                            uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
                    } catch (SecurityException ignored) {}
                }

                // Phase 4: cache and navigate
                BurstCache cache = new BurstCache();
                cache.clusters   = clusters;
                cache.uriStrings = urisToStrings(selectedUris);
                cache.scores     = scores;
                String json = new Gson().toJson(cache);

                File outFile = new File(getExternalFilesDir(null),
                    "burst_" + System.currentTimeMillis() + ".json");
                try (FileWriter fw = new FileWriter(outFile)) {
                    fw.write(json);
                }

                Intent intent = new Intent(this, BurstResultsActivity.class);
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

    private Bitmap decodeBitmap(Uri uri, int maxSide) throws IOException {
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inJustDecodeBounds = true;
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            BitmapFactory.decodeStream(is, null, opts);
        }
        opts.inSampleSize = MainActivity.computeSampleSize(opts.outWidth, opts.outHeight, maxSide);
        opts.inJustDecodeBounds = false;
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            Bitmap bmp = BitmapFactory.decodeStream(is, null, opts);
            if (bmp == null) throw new IOException("Could not decode image");
            return bmp;
        }
    }

    private static List<String> urisToStrings(List<Uri> uris) {
        List<String> out = new ArrayList<>(uris.size());
        for (Uri u : uris) out.add(u.toString());
        return out;
    }

    // --- Thumbnail strip adapter ---

    private class ThumbnailAdapter extends RecyclerView.Adapter<ThumbnailAdapter.VH> {

        private List<Uri> uris = new ArrayList<>();

        void setUris(List<Uri> uris) {
            this.uris = uris;
            notifyDataSetChanged();
        }

        @Override
        public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            ImageView iv = new ImageView(parent.getContext());
            int sizePx   = (int) (80 * getResources().getDisplayMetrics().density);
            int marginPx = (int) (2  * getResources().getDisplayMetrics().density);
            RecyclerView.LayoutParams lp = new RecyclerView.LayoutParams(sizePx, sizePx);
            lp.setMargins(marginPx, marginPx, marginPx, marginPx);
            iv.setLayoutParams(lp);
            iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
            return new VH(iv);
        }

        @Override
        public void onBindViewHolder(VH holder, int position) {
            Uri uri = uris.get(position);
            Executors.newSingleThreadExecutor().execute(() -> {
                try {
                    BitmapFactory.Options opts = new BitmapFactory.Options();
                    opts.inSampleSize = 4;
                    Bitmap bmp;
                    try (InputStream is = getContentResolver().openInputStream(uri)) {
                        bmp = BitmapFactory.decodeStream(is, null, opts);
                    }
                    if (bmp != null) runOnUiThread(() -> holder.iv.setImageBitmap(bmp));
                } catch (IOException ignored) {}
            });
        }

        @Override public int getItemCount() { return uris.size(); }

        class VH extends RecyclerView.ViewHolder {
            final ImageView iv;
            VH(ImageView iv) { super(iv); this.iv = iv; }
        }
    }

    // --- Cache POJO ---

    public static class BurstCache {
        public List<List<Integer>> clusters;
        public List<String>        uriStrings;
        public List<Float>         scores;
    }
}
