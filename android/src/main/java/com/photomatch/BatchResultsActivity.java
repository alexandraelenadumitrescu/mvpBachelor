package com.photomatch;

import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.view.ViewGroup;

import java.io.File;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.google.gson.Gson;
import com.photomatch.api.BatchResult;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executors;

public class BatchResultsActivity extends AppCompatActivity {

    static BatchActivity.BatchCache sharedCache;
    private BatchActivity.BatchCache cache;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_batch_results);

        String cachePath = getIntent().getStringExtra(BatchActivity.EXTRA_CACHE_PATH);
        if (cachePath == null) { finish(); return; }

        try (FileReader fr = new FileReader(cachePath)) {
            cache = new Gson().fromJson(fr, BatchActivity.BatchCache.class);
            sharedCache = cache;
        } catch (IOException e) {
            Toast.makeText(this, "Failed to load results", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        if (cache == null || cache.response == null) { finish(); return; }

        RecyclerView rv = findViewById(R.id.rvResults);
        rv.setLayoutManager(new GridLayoutManager(this, 2));
        rv.setAdapter(new ResultsAdapter());

        Button btnExport = findViewById(R.id.btnExport);
        btnExport.setOnClickListener(v -> exportAll());
    }

    private class ResultsAdapter extends RecyclerView.Adapter<ResultsAdapter.VH> {

        private final List<BatchResult> results = cache.response.results;
        private final List<String>      uriStrings = cache.originalUris;

        @Override
        public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            android.view.View view = getLayoutInflater()
                .inflate(R.layout.item_batch_result, parent, false);
            return new VH(view);
        }

        @Override
        public void onBindViewHolder(VH holder, int position) {
            BatchResult result = results.get(position);

            // Original thumbnail — load from URI with default Glide caching
            if (uriStrings != null && position < uriStrings.size()) {
                Uri uri = Uri.parse(uriStrings.get(position));
                Glide.with(BatchResultsActivity.this)
                    .load(uri)
                    .placeholder(new ColorDrawable(Color.parseColor("#1A1A1A")))
                    .centerCrop()
                    .into(holder.ivOriginal);
            }

            // Corrected thumbnail — prefer local file (v2), fall back to base64 (v1)
            Glide.with(BatchResultsActivity.this)
                .load(result.correctedPath != null ? new File(result.correctedPath)
                                                   : decodeBase64(result.correctedB64))
                .placeholder(new ColorDrawable(Color.parseColor("#1A1A1A")))
                .centerCrop()
                .into(holder.ivCorrected);

            holder.tvSimilarity.setText(
                String.format(Locale.US, "%d%%", Math.round(result.similarity * 100)));
            holder.tvRetrieved.setText(result.retrieved);

            boolean isBlurry = cache.blurFlags != null
                && position < cache.blurFlags.size()
                && Boolean.TRUE.equals(cache.blurFlags.get(position));
            holder.tvBlurBadge.setVisibility(isBlurry ? android.view.View.VISIBLE : android.view.View.GONE);

            holder.itemView.setOnClickListener(v -> openCompare(position, result));
        }

        @Override public int getItemCount() { return results.size(); }

        @Override
        public void onViewRecycled(VH holder) {
            super.onViewRecycled(holder);
            Glide.with(BatchResultsActivity.this).clear(holder.ivOriginal);
            Glide.with(BatchResultsActivity.this).clear(holder.ivCorrected);
        }

        class VH extends RecyclerView.ViewHolder {
            final ImageView ivOriginal;
            final ImageView ivCorrected;
            final TextView  tvSimilarity;
            final TextView  tvRetrieved;
            final TextView  tvBlurBadge;
            VH(android.view.View v) {
                super(v);
                ivOriginal   = v.findViewById(R.id.ivOriginal);
                ivCorrected  = v.findViewById(R.id.ivCorrected);
                tvSimilarity = v.findViewById(R.id.tvSimilarity);
                tvRetrieved  = v.findViewById(R.id.tvRetrieved);
                tvBlurBadge  = v.findViewById(R.id.tvBlurBadge);
            }
        }
    }

    private void openCompare(int position, BatchResult result) {
        Intent intent = new Intent(this, CompareActivity.class);
        intent.putExtra(CompareActivity.EXTRA_BATCH_INDEX, position);
        startActivity(intent);
    }

    private void exportAll() {
        if (cache.response.results.isEmpty()) {
            Toast.makeText(this, "No results to export", Toast.LENGTH_SHORT).show();
            return;
        }
        Executors.newSingleThreadExecutor().execute(() -> {
            int saved = 0;
            for (BatchResult result : cache.response.results) {
                try {
                    byte[] bytes = result.correctedPath != null
                        ? java.nio.file.Files.readAllBytes(new File(result.correctedPath).toPath())
                        : decodeBase64(result.correctedB64);
                    if (bytes != null) {
                        saveToGallery(bytes, result.retrieved);
                        saved++;
                    }
                } catch (IOException ignored) {}
            }
            final int finalSaved = saved;
            runOnUiThread(() ->
                Toast.makeText(this, "Saved " + finalSaved + " images to gallery",
                    Toast.LENGTH_SHORT).show());
        });
    }

    private void saveToGallery(byte[] imageBytes, String baseName) throws IOException {
        String filename = "photomatch_batch_" + baseName + "_" + System.currentTimeMillis() + ".jpg";
        ContentValues cv = new ContentValues();
        cv.put(MediaStore.Images.Media.DISPLAY_NAME, filename);
        cv.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            cv.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/PhotoMatch");
        } else {
            File dir = new File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                "PhotoMatch");
            //noinspection ResultOfMethodCallIgnored
            dir.mkdirs();
            cv.put(MediaStore.Images.Media.DATA, new File(dir, filename).getAbsolutePath());
        }

        Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv);
        if (uri == null) throw new IOException("MediaStore insert returned null");
        try (OutputStream os = getContentResolver().openOutputStream(uri)) {
            os.write(imageBytes);
        }
    }

    private static byte[] decodeBase64(String b64) {
        if (b64 == null) return null;
        return Base64.decode(b64, Base64.DEFAULT);
    }

}
