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
import com.photomatch.api.StyleUploadResponse;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Response;

public class StyleSetupActivity extends AppCompatActivity {

    private static final int   MAX_SLOTS   = 20;
    private static final String PREFS       = "photomatch_prefs";
    private static final String KEY_SESSION = "style_session_id";

    private final Uri[]        slotUris   = new Uri[MAX_SLOTS];
    private final ImageView[]  slotViews  = new ImageView[MAX_SLOTS];
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

        // Build 20 image slots programmatically
        int slotSizeDp = 80;
        int slotSizePx = (int) (slotSizeDp * getResources().getDisplayMetrics().density);
        int marginPx   = (int) (3 * getResources().getDisplayMetrics().density);

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
            iv.setOnClickListener(v -> {
                pendingSlot = slot;
                pickLauncher.launch("image/*");
            });

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
                if (bmp != null) {
                    final Bitmap finalBmp = bmp;
                    runOnUiThread(() -> iv.setImageBitmap(finalBmp));
                }
            } catch (IOException ignored) {}
        });
    }

    private void uploadStyle() {
        // Collect selected URIs
        List<Uri> selected = new ArrayList<>();
        for (Uri u : slotUris) {
            if (u != null) selected.add(u);
        }
        if (selected.isEmpty()) {
            Toast.makeText(this, "Select at least one photo", Toast.LENGTH_SHORT).show();
            return;
        }

        btnUpload.setEnabled(false);
        tvProgress.setVisibility(View.VISIBLE);
        tvProgress.setText("Preparing images...");

        executor = Executors.newSingleThreadExecutor();
        executor.execute(() -> {
            try {
                List<MultipartBody.Part> parts = new ArrayList<>();
                for (int i = 0; i < selected.size(); i++) {
                    final int idx = i + 1;
                    runOnUiThread(() ->
                        tvProgress.setText("Uploading " + idx + "/" + selected.size() + "..."));

                    File tmp = compressUri(selected.get(i), i);
                    RequestBody rb = RequestBody.create(MediaType.parse("image/jpeg"), tmp);
                    parts.add(MultipartBody.Part.createFormData("files", tmp.getName(), rb));
                }

                Response<StyleUploadResponse> resp = ApiClient.getInstance()
                    .getService().styleUpload(parts).execute();

                if (!resp.isSuccessful() || resp.body() == null) {
                    String err = resp.errorBody() != null ? resp.errorBody().string() : "unknown";
                    throw new IOException("Upload failed: HTTP " + resp.code() + " " + err);
                }

                String sessionId = resp.body().sessionId;
                getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                    .putString(KEY_SESSION, sessionId)
                    .apply();

                runOnUiThread(() -> {
                    Toast.makeText(this, "Style saved!", Toast.LENGTH_SHORT).show();
                    finish();
                });

            } catch (IOException e) {
                runOnUiThread(() -> {
                    tvProgress.setText("Error: " + e.getMessage());
                    btnUpload.setEnabled(true);
                });
            }
        });
    }

    private File compressUri(Uri uri, int index) throws IOException {
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inJustDecodeBounds = true;
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            BitmapFactory.decodeStream(is, null, opts);
        }
        opts.inSampleSize = MainActivity.computeSampleSize(opts.outWidth, opts.outHeight, 1200);
        opts.inJustDecodeBounds = false;

        Bitmap bmp;
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            bmp = BitmapFactory.decodeStream(is, null, opts);
        }
        if (bmp == null) throw new IOException("Could not decode image " + index);

        File out = new File(getCacheDir(), "style_" + index + ".jpg");
        try (FileOutputStream fos = new FileOutputStream(out)) {
            bmp.compress(Bitmap.CompressFormat.JPEG, 85, fos);
        }
        bmp.recycle();
        return out;
    }
}
