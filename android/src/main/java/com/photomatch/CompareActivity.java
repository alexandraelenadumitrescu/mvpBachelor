package com.photomatch;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Rect;
import android.graphics.drawable.ColorDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.ViewTreeObserver;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;
import com.google.gson.Gson;
import com.photomatch.api.BatchResult;
import com.photomatch.db.AppDatabase;
import com.photomatch.db.FavoriteDao;
import com.photomatch.db.FavoritePhoto;

import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CompareActivity extends AppCompatActivity {

    static final String EXTRA_ORIGINAL_URI  = "original_uri";
    static final String EXTRA_CORRECTED_B64 = "corrected_b64";
    static final String EXTRA_SIMILARITY    = "similarity";
    static final String EXTRA_RETRIEVED     = "retrieved";
    static final String EXTRA_BATCH_INDEX   = "batch_item_index";

    private FrameLayout    flCompare;
    private ImageView      ivBefore;
    private android.view.View vDivider;
    private ImageButton    btnStar;
    private HistogramView  histogramBefore;
    private HistogramView  histogramAfter;

    private float  dividerFraction      = 0.5f;
    private int    favoriteId           = -1;
    private float  matchAestheticScore  = 0f;

    private String retrieved;
    private String correctedB64;
    private String correctedPath;   // v2: local file path (preferred over correctedB64)
    private String originalUri;
    private float  similarity;
    private ExecutorService executor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_compare);

        // Resolve data — from batch static cache (avoids Binder limit) or from Intent extras
        int batchIndex = getIntent().getIntExtra(EXTRA_BATCH_INDEX, -1);
        if (batchIndex >= 0 && BatchResultsActivity.sharedCache != null) {
            BatchActivity.BatchCache bc = BatchResultsActivity.sharedCache;
            BatchResult r = bc.response.results.get(batchIndex);
            correctedB64        = r.correctedB64;
            correctedPath       = r.correctedPath;
            similarity          = r.similarity;
            retrieved           = r.retrieved;
            matchAestheticScore = r.matchAestheticScore;
            int originalPhotoIdx = r.index;
            originalUri = (bc.originalUris != null && originalPhotoIdx >= 0
                           && originalPhotoIdx < bc.originalUris.size())
                          ? bc.originalUris.get(originalPhotoIdx) : null;
        } else {
            originalUri  = getIntent().getStringExtra(EXTRA_ORIGINAL_URI);
            correctedB64 = getIntent().getStringExtra(EXTRA_CORRECTED_B64);
            similarity   = getIntent().getFloatExtra(EXTRA_SIMILARITY, 0f);
            retrieved    = getIntent().getStringExtra(EXTRA_RETRIEVED);
        }

        if (correctedB64 == null) { finish(); return; }

        executor = Executors.newSingleThreadExecutor();

        flCompare        = findViewById(R.id.flCompare);
        ivBefore         = findViewById(R.id.ivBefore);
        vDivider         = findViewById(R.id.vDivider);
        ImageView ivAfter     = findViewById(R.id.ivAfter);
        TextView  tvMatchInfo = findViewById(R.id.tvMatchInfo);
        btnStar               = findViewById(R.id.btnStar);
        histogramBefore  = findViewById(R.id.histogramBefore);
        histogramAfter   = findViewById(R.id.histogramAfter);

        // Load images via Glide
        if (originalUri != null) {
            Glide.with(this)
                .load(Uri.parse(originalUri))
                .placeholder(new ColorDrawable(Color.parseColor("#1A1A1A")))
                .fitCenter()
                .into(ivBefore);
        } else {
            ivBefore.setImageDrawable(new ColorDrawable(Color.parseColor("#1A1A1A")));
        }
        if (correctedPath != null) {
            Glide.with(this).load(new java.io.File(correctedPath)).fitCenter().into(ivAfter);
        } else {
            Glide.with(this).load(decodeBase64(correctedB64)).fitCenter().into(ivAfter);
        }

        int similarityPct = Math.round(similarity * 100);
        tvMatchInfo.setText(String.format(Locale.US,
            "MATCHED  %s  ·  %d%%  ·  AES %.2f",
            retrieved != null ? retrieved : "",
            similarityPct,
            matchAestheticScore));

        // Compute RGB histograms on background thread
        final String capturedUri = originalUri;
        final String capturedB64 = correctedB64;
        executor.execute(() -> {
            float[][] histBefore = null, histAfter = null;
            if (capturedUri != null) {
                try (InputStream is = getContentResolver().openInputStream(Uri.parse(capturedUri))) {
                    Bitmap b = BitmapFactory.decodeStream(is);
                    if (b != null) { histBefore = HistogramUtils.compute(b); b.recycle(); }
                } catch (IOException ignored) {}
            }
            Bitmap afterBmp = null;
            if (correctedPath != null) {
                afterBmp = BitmapFactory.decodeFile(correctedPath);
            } else if (capturedB64 != null) {
                byte[] b = decodeBase64(capturedB64);
                if (b != null) afterBmp = BitmapFactory.decodeByteArray(b, 0, b.length);
            }
            if (afterBmp != null) { histAfter = HistogramUtils.compute(afterBmp); afterBmp.recycle(); }
            final float[][] fBefore = histBefore, fAfter = histAfter;
            runOnUiThread(() -> {
                if (fBefore != null) histogramBefore.setData(fBefore, "Before");
                if (fAfter  != null) histogramAfter .setData(fAfter,  "After");
            });
        });

        // Draggable divider
        flCompare.setOnTouchListener((v, event) -> {
            int action = event.getAction();
            if (action == MotionEvent.ACTION_DOWN || action == MotionEvent.ACTION_MOVE) {
                int width = flCompare.getWidth();
                if (width > 0) {
                    dividerFraction = Math.max(0.05f, Math.min(0.95f, event.getX() / width));
                    updateDivider();
                }
                return true;
            }
            return false;
        });

        flCompare.getViewTreeObserver().addOnGlobalLayoutListener(
            new ViewTreeObserver.OnGlobalLayoutListener() {
                @Override
                public void onGlobalLayout() {
                    flCompare.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                    updateDivider();
                }
            });

        // Star button — check DB state
        executor.execute(() -> {
            FavoritePhoto existing = retrieved != null
                ? AppDatabase.get(this).favoriteDao().findByRetrieved(retrieved)
                : null;
            if (existing != null) favoriteId = existing.id;
            runOnUiThread(this::updateStarButton);
        });

        btnStar.setOnClickListener(v -> toggleFavorite());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor != null) executor.shutdown();
    }

    private void updateDivider() {
        int width  = flCompare.getWidth();
        int height = flCompare.getHeight();
        if (width == 0 || height == 0) return;
        int divX = (int) (dividerFraction * width);
        vDivider.setTranslationX(divX - vDivider.getWidth() / 2f);
        ivBefore.setClipBounds(new Rect(0, 0, divX, height));
        ivBefore.invalidate();
    }

    private void updateStarButton() {
        btnStar.setImageResource(favoriteId != -1
            ? R.drawable.ic_star_filled : R.drawable.ic_star_outline);
    }

    private void toggleFavorite() {
        btnStar.setEnabled(false);
        executor.execute(() -> {
            FavoriteDao dao = AppDatabase.get(this).favoriteDao();
            if (favoriteId != -1) {
                FavoritePhoto f = new FavoritePhoto();
                f.id = favoriteId;
                dao.delete(f);
                favoriteId = -1;
            } else {
                FavoritePhoto f = buildFavorite();
                dao.insert(f);
                FavoritePhoto inserted = dao.findByRetrieved(retrieved);
                favoriteId = inserted != null ? inserted.id : -1;
            }
            final boolean added = favoriteId != -1;
            runOnUiThread(() -> {
                updateStarButton();
                btnStar.setEnabled(true);
                Toast.makeText(this,
                    added ? "Added to favorites" : "Removed from favorites",
                    Toast.LENGTH_SHORT).show();
            });
        });
    }

    private FavoritePhoto buildFavorite() {
        Map<String, Object> imp = new LinkedHashMap<>();
        imp.put("similarity", similarity);
        FavoritePhoto f = new FavoritePhoto();
        // v2: corrected image is a file; store null in Room (image is on disk already)
        f.correctedBase64 = correctedPath != null ? null : correctedB64;
        f.retrieved       = retrieved;
        f.timestamp       = System.currentTimeMillis();
        f.improvements    = new Gson().toJson(imp);
        return f;
    }

    private static byte[] decodeBase64(String b64) {
        if (b64 == null) return null;
        return android.util.Base64.decode(b64, android.util.Base64.DEFAULT);
    }
}
