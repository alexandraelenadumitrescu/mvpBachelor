package com.photomatch;

import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
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
import androidx.appcompat.app.AppCompatActivity;

import com.photomatch.api.ApiClient;
import com.photomatch.api.StyleVectorsRequest;
import com.photomatch.api.StyleVectorsResponse;
import com.photomatch.ml.CLIPEncoder;
import com.photomatch.ml.DefectDetector;
import com.photomatch.ml.HybridVectorBuilder;

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
 * Computes 517-dim hybrid vectors locally for each style photo,
 * then sends only the vectors to /style/vectors.
 */
public class StyleSetupActivity extends AppCompatActivity {

    private static final int    MAX_SLOTS   = 20;
    private static final String PREFS       = "photomatch_prefs";
    private static final String KEY_SESSION = "style_session_id";

    private final Uri[]        slotUris  = new Uri[MAX_SLOTS];
    private final ImageView[]  slotViews = new ImageView[MAX_SLOTS];
    private int                pendingSlot = -1;

    private Button   btnUpload;
    private TextView tvProgress;
    private ExecutorService executor;

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

        btnUpload  = findViewById(R.id.btnUpload);
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

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor != null) executor.shutdownNow();
    }

    private void loadThumbnail(ImageView iv, Uri uri) {
        Executors.newSingleThreadExecutor().execute(() -> {
            try {
                BitmapFactory.Options opts = new BitmapFactory.Options();
                opts.inJustDecodeBounds = true;
                try (InputStream is = getContentResolver().openInputStream(uri)) {
                    BitmapFactory.decodeStream(is, null, opts);
                }
                opts.inSampleSize = MainActivity.computeSampleSize(
                    opts.outWidth, opts.outHeight, 200);
                opts.inJustDecodeBounds = false;
                Bitmap bmp;
                try (InputStream is = getContentResolver().openInputStream(uri)) {
                    bmp = BitmapFactory.decodeStream(is, null, opts);
                }
                if (bmp != null) { final Bitmap b = bmp; runOnUiThread(() -> iv.setImageBitmap(b)); }
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

        btnUpload.setEnabled(false);
        tvProgress.setVisibility(View.VISIBLE);
        tvProgress.setText("Loading CLIP model...");

        executor = Executors.newSingleThreadExecutor();
        executor.execute(() -> {
            try {
                // Init CLIP encoder
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
                    final int idx = i + 1;
                    final int total = selected.size();
                    runOnUiThread(() ->
                        tvProgress.setText("Computing vectors " + idx + "/" + total + "..."));

                    Bitmap bmp = decodeBitmap(selected.get(i), 512);
                    if (bmp == null) continue;

                    float[] clip   = clipEncoder.encode(bmp);
                    float[] defect = defectDetector.detect(bmp);
                    bmp.recycle();
                    vectors.add(HybridVectorBuilder.build(clip, defect));
                }

                clipEncoder.close();
                defectDetector.close();

                if (vectors.isEmpty()) throw new IOException("No images could be encoded");

                runOnUiThread(() -> tvProgress.setText("Sending vectors to server..."));

                // Read existing session_id (null → server creates new one)
                String existingSid = getSharedPreferences(PREFS, MODE_PRIVATE)
                    .getString(KEY_SESSION, null);

                Response<StyleVectorsResponse> resp = ApiClient.getInstance()
                    .getService().styleVectors(new StyleVectorsRequest(vectors, existingSid))
                    .execute();

                if (!resp.isSuccessful() || resp.body() == null) {
                    String err = resp.errorBody() != null ? resp.errorBody().string() : "";
                    throw new IOException("Failed: HTTP " + resp.code() + " " + err);
                }

                String sessionId = resp.body().sessionId;
                getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                    .putString(KEY_SESSION, sessionId).apply();

                runOnUiThread(() -> {
                    Toast.makeText(this, "Style saved!", Toast.LENGTH_SHORT).show();
                    finish();
                });

            } catch (Exception e) {
                runOnUiThread(() -> {
                    tvProgress.setText("Error: " + e.getMessage());
                    btnUpload.setEnabled(true);
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
}
