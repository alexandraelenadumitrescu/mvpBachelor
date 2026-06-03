package com.photomatch;

import android.content.ContentValues;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;
import com.google.gson.Gson;
import com.photomatch.api.ProcessResponse;
import com.photomatch.db.AppDatabase;
import com.photomatch.db.FavoriteDao;
import com.photomatch.db.FavoritePhoto;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ResultsActivity extends AppCompatActivity {

    private static final String FILL  = "████████";
    private static final String EMPTY = "░░░░░░░░";

    private static final String[] DEFECT_KEYS   = {"blur", "noise", "overexposure", "underexposure", "compression"};
    private static final String[] DEFECT_LABELS = {"BLUR ", "NOISE", "OVER ", "UNDER", "COMP "};

    private int             favoriteId = -1;
    private ImageButton     btnHeart;
    private ExecutorService executor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_results);

        ProcessResponse resp = ResponseCache.current;
        if (resp == null) { finish(); return; }

        executor = Executors.newSingleThreadExecutor();

        ImageView     ivResult       = findViewById(R.id.ivResult);
        TextView      tvDefects      = findViewById(R.id.tvDefects);
        btnHeart                     = (ImageButton) findViewById(R.id.btnHeart);
        Button        btnMatch       = findViewById(R.id.btnViewMatch);
        Button        btnSave        = findViewById(R.id.btnSave);
        HistogramView histogramResult = findViewById(R.id.histogramResult);

        // Show corrected image via Glide (manages bitmap lifecycle)
        Glide.with(this).load(decodeBase64(resp.finalB64)).fitCenter().into(ivResult);

        // Compute histogram on background thread
        executor.execute(() -> {
            byte[] bytes = decodeBase64(resp.finalB64);
            if (bytes == null) return;
            android.graphics.Bitmap b =
                android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
            if (b == null) return;
            float[][] hist = HistogramUtils.compute(b);
            b.recycle();
            runOnUiThread(() -> histogramResult.setData(hist, null));
        });

        // Defect bars
        tvDefects.setText(buildDefectBars(resp.defects));

        // Match quality line
        TextView tvMatchQuality = findViewById(R.id.tvMatchQuality);
        tvMatchQuality.setText(String.format(Locale.US,
            "Similarity  %.2f  ·  Aesthetic  %.2f",
            resp.similarity, resp.matchAestheticScore));

        // Check if already favorited
        executor.execute(() -> {
            FavoritePhoto existing = AppDatabase.get(this).favoriteDao().findByRetrieved(resp.retrieved);
            if (existing != null) favoriteId = existing.id;
            runOnUiThread(this::updateHeartButton);
        });

        btnHeart.setOnClickListener(v -> toggleFavorite(resp));

        btnMatch.setOnClickListener(v ->
            startActivity(new Intent(this, DetailActivity.class)));

        btnSave.setOnClickListener(v -> saveToGallery(decodeBase64(resp.finalB64)));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor != null) executor.shutdown();
    }

    private void toggleFavorite(ProcessResponse resp) {
        executor.execute(() -> {
            FavoriteDao dao = AppDatabase.get(this).favoriteDao();
            if (favoriteId != -1) {
                FavoritePhoto f = new FavoritePhoto();
                f.id = favoriteId;
                dao.delete(f);
                favoriteId = -1;
            } else {
                FavoritePhoto f = buildFavorite(resp);
                dao.insert(f);
                FavoritePhoto inserted = dao.findByRetrieved(resp.retrieved);
                favoriteId = inserted != null ? inserted.id : -1;
            }
            final boolean added = favoriteId != -1;
            runOnUiThread(() -> {
                updateHeartButton();
                Toast.makeText(this,
                    added ? "Added to favorites" : "Removed from favorites",
                    Toast.LENGTH_SHORT).show();
            });
        });
    }

    private void updateHeartButton() {
        btnHeart.setImageResource(favoriteId != -1
            ? R.drawable.ic_star_filled : R.drawable.ic_star_outline);
    }

    private FavoritePhoto buildFavorite(ProcessResponse resp) {
        boolean lutApplied = resp.note != null && resp.note.isEmpty();
        Map<String, Object> imp = new LinkedHashMap<>();
        if (resp.defects != null) imp.putAll(resp.defects);
        imp.put("retrieved", resp.retrieved);
        imp.put("similarity", resp.similarity);
        imp.put("lut_applied", lutApplied);
        FavoritePhoto f = new FavoritePhoto();
        f.originalBase64  = resp.originalB64;
        f.correctedBase64 = resp.finalB64;
        f.retrieved       = resp.retrieved;
        f.timestamp       = System.currentTimeMillis();
        f.improvements    = new Gson().toJson(imp);
        return f;
    }

    private String buildDefectBars(Map<String, Float> defects) {
        if (defects == null) return "No defect data";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < DEFECT_KEYS.length; i++) {
            float val = defects.containsKey(DEFECT_KEYS[i]) ? defects.get(DEFECT_KEYS[i]) : 0f;
            int filled = Math.min(8, Math.max(0, Math.round(val * 8)));
            String bar = FILL.substring(0, filled) + EMPTY.substring(0, 8 - filled);
            sb.append(String.format(Locale.US, "%s %s %.2f\n", DEFECT_LABELS[i], bar, val));
        }
        if (sb.length() > 0) sb.setLength(sb.length() - 1);
        return sb.toString();
    }

    private void saveToGallery(byte[] imageBytes) {
        if (imageBytes == null) {
            Toast.makeText(this, "No image to save", Toast.LENGTH_SHORT).show();
            return;
        }
        try {
            String filename = "photomatch_" + System.currentTimeMillis() + ".jpg";
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
                cv.put(MediaStore.Images.Media.DATA,
                    new File(dir, filename).getAbsolutePath());
            }

            Uri uri = getContentResolver().insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv);
            if (uri == null) throw new IOException("MediaStore insert returned null");

            try (OutputStream os = getContentResolver().openOutputStream(uri)) {
                os.write(imageBytes);
            }
            Toast.makeText(this, "Saved to gallery", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            Toast.makeText(this, "Save failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private static byte[] decodeBase64(String b64) {
        if (b64 == null) return null;
        return android.util.Base64.decode(b64, android.util.Base64.DEFAULT);
    }
}
