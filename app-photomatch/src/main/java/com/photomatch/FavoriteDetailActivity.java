package com.photomatch;

import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.photomatch.db.AppDatabase;
import com.photomatch.db.FavoritePhoto;

import java.lang.reflect.Type;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executors;

public class FavoriteDetailActivity extends AppCompatActivity {

    private static final String FILL  = "████████";
    private static final String EMPTY = "░░░░░░░░";

    private FrameLayout flCompare;
    private ImageView   ivBefore;
    private View        vDivider;

    private float dividerFraction = 0.5f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_favorite_detail);

        FavoritePhoto fav = FavoritesActivity.selectedFavorite;
        if (fav == null) { finish(); return; }

        flCompare = findViewById(R.id.flCompare);
        ivBefore  = findViewById(R.id.ivBefore);
        vDivider  = findViewById(R.id.vDivider);
        ImageView ivAfter          = findViewById(R.id.ivAfter);
        TextView  tvMatchInfo      = findViewById(R.id.tvMatchInfo);
        TextView  tvImprovements   = findViewById(R.id.tvImprovements);
        Button    btnUnfavorite    = findViewById(R.id.btnUnfavorite);

        if (fav.originalBase64 == null) {
            // Burst photo: no before/after, show image full-screen via URI or correctedBase64
            flCompare.setVisibility(View.GONE);
            // Re-use ivAfter as a standalone full-screen image in the layout
            // We reveal it by showing the compare frame partially — actually just load via uri
        } else {
            // Server result: load before/after
            Glide.with(this).load(decodeBase64(fav.originalBase64)).fitCenter().into(ivBefore);
            Glide.with(this).load(decodeBase64(fav.correctedBase64)).fitCenter().into(ivAfter);

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
        }

        // Match info + improvements
        tvMatchInfo.setText(buildMatchInfo(fav));
        tvImprovements.setText(buildImprovements(fav.improvements));

        btnUnfavorite.setOnClickListener(v -> removeFavorite(fav));
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

    private void removeFavorite(FavoritePhoto fav) {
        Executors.newSingleThreadExecutor().execute(() -> {
            AppDatabase.get(this).favoriteDao().delete(fav);
            runOnUiThread(() -> {
                Toast.makeText(this, "Removed from favorites", Toast.LENGTH_SHORT).show();
                finish();
            });
        });
    }

    private String buildMatchInfo(FavoritePhoto fav) {
        if (fav.retrieved == null || fav.retrieved.isEmpty()
                || fav.retrieved.startsWith("content://")) {
            return "BURST PHOTO";
        }
        // Parse similarity from improvements JSON
        float similarity = 0f;
        try {
            Type mapType = new TypeToken<Map<String, Object>>(){}.getType();
            Map<String, Object> imp = new Gson().fromJson(fav.improvements, mapType);
            if (imp != null && imp.containsKey("similarity")) {
                similarity = ((Number) imp.get("similarity")).floatValue();
            }
        } catch (Exception ignored) {}
        return String.format(Locale.US, "MATCHED  %s  ·  %d%% SIMILAR",
            fav.retrieved, Math.round(similarity * 100));
    }

    @SuppressWarnings("unchecked")
    private String buildImprovements(String json) {
        if (json == null) return "";
        try {
            Type mapType = new TypeToken<Map<String, Object>>(){}.getType();
            Map<String, Object> imp = new Gson().fromJson(json, mapType);
            if (imp == null) return "";

            StringBuilder sb = new StringBuilder();

            String[] defectKeys   = {"blur", "noise", "overexposure", "underexposure", "compression"};
            String[] defectLabels = {"BLUR CORR  ", "NOISE CORR ", "OVEREXP    ", "UNDEREXP   ", "COMPRESS   "};

            for (int i = 0; i < defectKeys.length; i++) {
                Object val = imp.get(defectKeys[i]);
                if (val != null) {
                    float f = ((Number) val).floatValue();
                    int pct = Math.round(f * 100);
                    int filled = Math.min(8, Math.max(0, Math.round(f * 8)));
                    String bar = FILL.substring(0, filled) + EMPTY.substring(0, 8 - filled);
                    sb.append(String.format(Locale.US, "%s %s %d%%\n", defectLabels[i], bar, pct));
                }
            }

            Object lutApplied = imp.get("lut_applied");
            if (lutApplied != null) {
                boolean lut = Boolean.TRUE.equals(lutApplied);
                sb.append("LUT APPLIED           ").append(lut ? "YES" : "NO (CLAHE ONLY)").append("\n");
            }

            Object aestheticScore = imp.get("aesthetic_score");
            if (aestheticScore != null) {
                float f = ((Number) aestheticScore).floatValue();
                sb.append(String.format(Locale.US, "AESTHETIC SCORE       %.2f\n", f));
            }

            if (sb.length() > 0) sb.setLength(sb.length() - 1);
            return sb.toString();
        } catch (Exception e) {
            return "";
        }
    }

    private static byte[] decodeBase64(String b64) {
        if (b64 == null) return null;
        return android.util.Base64.decode(b64, android.util.Base64.DEFAULT);
    }
}
