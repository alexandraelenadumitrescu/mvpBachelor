package com.photomatch;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.GridLayout;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;

import com.photomatch.api.StyleVectorsRequest;
import com.photomatch.api.StyleVectorsResponse;
import com.photomatch.base.BaseServerActivity;
import com.photomatch.ml.CLIPEncoder;
import com.photomatch.ml.DefectDetector;
import com.photomatch.ml.HybridVectorBuilder;
import com.photomatch.util.ImageUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;

import retrofit2.Response;

public class StyleSetupActivity extends BaseServerActivity {

    private static final int    MAX_SLOTS   = 20;
    private static final String KEY_SESSION = "style_session_id";

    private final Uri[]       slotUris  = new Uri[MAX_SLOTS];
    private final ImageView[] slotViews = new ImageView[MAX_SLOTS];
    private int               pendingSlot = -1;

    private Button   btnUpload;
    private TextView tvProgress;

    private final ActivityResultLauncher<String> pickLauncher =
        registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
            if (uri != null && pendingSlot >= 0) {
                slotUris[pendingSlot] = uri;
                loadThumbnail(slotViews[pendingSlot], uri);
                pendingSlot = -1;
            }
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_style_setup);

        btnUpload  = primaryButton = findViewById(R.id.btnUpload);
        tvProgress = findViewById(R.id.tvProgress);
        GridLayout glSlots = findViewById(R.id.glSlots);

        int slotSizePx = (int) (80 * getResources().getDisplayMetrics().density);
        int marginPx   = (int) (3  * getResources().getDisplayMetrics().density);

        for (int i = 0; i < MAX_SLOTS; i++) {
            ImageView iv = new ImageView(this);
            iv.setBackgroundColor(Color.parseColor("#1A1A1A"));
            iv.setScaleType(ImageView.ScaleType.CENTER_CROP);

            GridLayout.LayoutParams lp = new GridLayout.LayoutParams();
            lp.width  = slotSizePx;
            lp.height = slotSizePx;
            lp.setMargins(marginPx, marginPx, marginPx, marginPx);
            lp.columnSpec = GridLayout.spec(i % 4, 1f);
            lp.rowSpec    = GridLayout.spec(i / 4, 1f);
            lp.setGravity(Gravity.FILL);
            iv.setLayoutParams(lp);

            final int slot = i;
            iv.setOnClickListener(v -> { pendingSlot = slot; pickLauncher.launch("image/*"); });
            slotViews[i] = iv;
            glSlots.addView(iv);
        }

        btnUpload.setOnClickListener(v -> uploadStyle());
    }

    private void loadThumbnail(ImageView iv, Uri uri) {
        Executors.newSingleThreadExecutor().execute(() -> {
            try {
                Bitmap bmp = ImageUtils.decodeBitmap(getContentResolver(), uri, 200);
                runOnUiThread(() -> iv.setImageBitmap(bmp));
            } catch (IOException ignored) {}
        });
    }

    private void uploadStyle() {
        List<Uri> selected = new ArrayList<>();
        for (Uri u : slotUris) if (u != null) selected.add(u);
        if (selected.isEmpty()) {
            Toast.makeText(this, "Select at least one photo", Toast.LENGTH_SHORT).show();
            return;
        }

        setProcessing(true);
        tvProgress.setVisibility(View.VISIBLE);
        tvProgress.setText("Loading CLIP model...");

        executor.execute(() -> {
            try {
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
                List<float[]> vectors = new ArrayList<>();

                for (int i = 0; i < selected.size(); i++) {
                    final int idx = i + 1, total = selected.size();
                    runOnUiThread(() -> tvProgress.setText("Computing vectors " + idx + "/" + total + "..."));

                    Bitmap bmp = ImageUtils.decodeBitmap(getContentResolver(), selected.get(i), 512);
                    float[] clip   = clipEncoder.encode(bmp);
                    float[] defect = defectDetector.detect(bmp);
                    bmp.recycle();
                    vectors.add(HybridVectorBuilder.build(clip, defect));
                }

                clipEncoder.close();
                defectDetector.close();

                if (vectors.isEmpty()) throw new IOException("No images could be encoded");

                runOnUiThread(() -> tvProgress.setText("Sending vectors to server..."));

                String existingSid = getSharedPreferences(
                    com.photomatch.api.ApiClient.PREFS_NAME, MODE_PRIVATE).getString(KEY_SESSION, null);

                Response<StyleVectorsResponse> resp =
                    api().styleVectors(new StyleVectorsRequest(vectors, existingSid)).execute();

                if (!resp.isSuccessful() || resp.body() == null) {
                    String err = resp.errorBody() != null ? resp.errorBody().string() : "";
                    throw new IOException("Failed: HTTP " + resp.code() + " " + err);
                }

                getSharedPreferences(com.photomatch.api.ApiClient.PREFS_NAME, MODE_PRIVATE)
                    .edit().putString(KEY_SESSION, resp.body().sessionId).apply();

                runOnUiThread(() -> {
                    Toast.makeText(this, "Style saved!", Toast.LENGTH_SHORT).show();
                    finish();
                });

            } catch (Exception e) {
                runOnUiThread(() -> {
                    tvProgress.setText("Error: " + e.getMessage());
                    setProcessing(false);
                });
            }
        });
    }
}
