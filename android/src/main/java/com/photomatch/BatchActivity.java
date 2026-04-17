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
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.gson.Gson;
import com.photomatch.api.ApiClient;
import com.photomatch.api.BatchResponse;
import com.photomatch.api.BatchResult;
import com.photomatch.api.SearchAndCorrectRequest;
import com.photomatch.api.SearchAndCorrectResponse;
import com.photomatch.ml.CLIPEncoder;
import com.photomatch.ml.DefectDetector;
import com.photomatch.ml.HybridVectorBuilder;
import com.photomatch.ml.ImageCorrector;
import com.photomatch.ml.LutCache;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import retrofit2.Response;

/**
 * V2 — images never leave the device.
 * Per-photo: on-device CLIP + defect → /search_and_correct (vector) → LUT download →
 * ImageCorrector → BatchResultsActivity.
 */
public class BatchActivity extends AppCompatActivity {

    private static final int MAX_PHOTOS = 100;
    static final String EXTRA_CACHE_PATH = "cache_path";

    private List<Uri> selectedUris = new ArrayList<>();

    private Button       btnPick;
    private Button       btnProcess;
    private TextView     tvCount;
    private TextView     tvProgress;
    private TextView     tvError;
    private ProgressBar  progressBar;
    private RecyclerView rvThumbnails;
    private ThumbnailAdapter thumbnailAdapter;
    private ExecutorService  executor;

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
        btnProcess   = findViewById(R.id.btnProcess);
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
        btnProcess.setOnClickListener(v -> startBatchProcessing());
        executor = Executors.newSingleThreadExecutor();
        updateProcessButton();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor != null) executor.shutdownNow();
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

    private void showError(String msg) {
        tvError.setText(msg);
        tvError.setVisibility(View.VISIBLE);
    }

    private void setUiProcessing(boolean processing) {
        btnPick.setEnabled(!processing);
        btnProcess.setEnabled(!processing);
        tvProgress.setVisibility(processing ? View.VISIBLE : View.GONE);
        progressBar.setVisibility(processing ? View.VISIBLE : View.GONE);
        tvError.setVisibility(View.GONE);
    }

    private void startBatchProcessing() {
        if (selectedUris.isEmpty()) return;
        setUiProcessing(true);
        progressBar.setMax(selectedUris.size());
        progressBar.setIndeterminate(false);

        executor.execute(() -> {
            try {
                // Init models once, reuse across all photos
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

                List<BatchResult>  results   = new ArrayList<>();
                List<Boolean>      blurFlags = new ArrayList<>();
                int failed = 0;

                for (int i = 0; i < selectedUris.size(); i++) {
                    final int idx = i;
                    final int total = selectedUris.size();
                    runOnUiThread(() -> {
                        tvProgress.setText("Processing " + (idx + 1) + "/" + total + "...");
                        progressBar.setProgress(idx + 1);
                    });

                    Uri uri = selectedUris.get(i);
                    try {
                        // 1. Decode once — reuse same bitmap for CLIP, defect, and correction
                        Bitmap bmp = decodeBitmap(uri, 1200);
                        if (bmp == null) { failed++; blurFlags.add(false); continue; }

                        blurFlags.add(BlurDetector.check(bmp).isBlurry);

                        float[] clip   = clipEncoder.encode(bmp);
                        float[] defect = defectDetector.detect(bmp);
                        float[] hybrid = HybridVectorBuilder.build(clip, defect);

                        // 2. FAISS retrieval — vector only
                        Response<SearchAndCorrectResponse> searchResp =
                            ApiClient.getInstance().getService()
                                .searchAndCorrect(new SearchAndCorrectRequest(hybrid), 0.3f, false)
                                .execute();
                        if (!searchResp.isSuccessful() || searchResp.body() == null) {
                            bmp.recycle(); failed++; blurFlags.set(blurFlags.size()-1, false); continue;
                        }
                        SearchAndCorrectResponse body = searchResp.body();

                        // 3. LUT download + on-device correction
                        float[] lut = LutCache.get(this, body.retrieved);
                        Bitmap corrected = ImageCorrector.correct(bmp, lut);
                        bmp.recycle();

                        // 4. Save to temp file — avoids base64 overhead and giant JSON
                        File outFile = saveCorrectedToFile(corrected, i);
                        corrected.recycle();

                        BatchResult result = new BatchResult();
                        result.index                = i;
                        result.retrieved            = body.retrieved;
                        result.similarity           = body.similarity;
                        result.correctedPath        = outFile != null ? outFile.getAbsolutePath() : null;
                        result.matchAestheticScore  = body.matchAestheticScore;
                        results.add(result);

                    } catch (Exception e) {
                        failed++;
                        if (blurFlags.size() <= i) blurFlags.add(false);
                    }
                }

                clipEncoder.close();
                defectDetector.close();

                // Build BatchResponse to reuse BatchResultsActivity
                BatchResponse batchResponse = new BatchResponse();
                batchResponse.results   = results;
                batchResponse.processed = results.size();
                batchResponse.failed    = failed;

                BatchCache cache = new BatchCache();
                cache.response     = batchResponse;
                cache.originalUris = urisToStrings(selectedUris);
                cache.blurFlags    = blurFlags;
                String json = new Gson().toJson(cache);

                File outFile = new File(getExternalFilesDir(null),
                    "batch_" + System.currentTimeMillis() + ".json");
                try (FileWriter fw = new FileWriter(outFile)) { fw.write(json); }

                Intent intent = new Intent(this, BatchResultsActivity.class);
                intent.putExtra(EXTRA_CACHE_PATH, outFile.getAbsolutePath());
                runOnUiThread(() -> { startActivity(intent); finish(); });

            } catch (Exception e) {
                runOnUiThread(() -> { setUiProcessing(false); showError("Error: " + e.getMessage()); });
            }
        });
    }

    // ── helpers ────────────────────────────────────────────────────────────────

    private Bitmap decodeBitmap(Uri uri, int maxSide) throws IOException {
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inJustDecodeBounds = true;
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            BitmapFactory.decodeStream(is, null, opts);
        }
        opts.inSampleSize = MainActivity.computeSampleSize(opts.outWidth, opts.outHeight, maxSide);
        opts.inJustDecodeBounds = false;
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            return BitmapFactory.decodeStream(is, null, opts);
        }
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

    // ── Thumbnail adapter ─────────────────────────────────────────────────────

    private class ThumbnailAdapter extends RecyclerView.Adapter<ThumbnailAdapter.VH> {
        private List<Uri> uris = new ArrayList<>();
        void setUris(List<Uri> uris) { this.uris = uris; notifyDataSetChanged(); }

        @Override public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            ImageView iv = new ImageView(parent.getContext());
            int sz = (int)(80 * getResources().getDisplayMetrics().density);
            int mg = (int)(2  * getResources().getDisplayMetrics().density);
            RecyclerView.LayoutParams lp = new RecyclerView.LayoutParams(sz, sz);
            lp.setMargins(mg, mg, mg, mg);
            iv.setLayoutParams(lp);
            iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
            return new VH(iv);
        }
        @Override public void onBindViewHolder(VH holder, int position) {
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

    // ── Cache POJO ────────────────────────────────────────────────────────────

    public static class BatchCache {
        public BatchResponse response;
        public List<String>  originalUris;
        public List<Boolean> blurFlags;
    }
}
