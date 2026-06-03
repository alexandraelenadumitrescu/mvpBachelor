package com.photomatch;

import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.documentfile.provider.DocumentFile;

import com.photomatch.api.BlurDetectResponse;
import com.photomatch.api.BlurRegion;
import com.photomatch.base.BaseServerActivity;
import com.photomatch.ui.ZoomableImageView;
import com.photomatch.util.ImageUtils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Response;

public class BlurActivity extends BaseServerActivity {

    // Image state
    private final List<Uri>          imageUris    = new ArrayList<>();
    private final Map<Integer, Bitmap> origBoxed   = new HashMap<>(); // original + drawn boxes
    private final Map<Integer, Bitmap> blurred     = new HashMap<>(); // blurred result
    private int  currentIndex  = 0;
    private boolean showBlurred = false;

    // Views
    private ZoomableImageView ivMain;
    private Button            btnToggle;
    private TextView          tvRegions;
    private TextView          tvCounter;
    private LinearLayout      navigationRow;
    private Button            btnPrev;
    private Button            btnNext;
    private Switch            switchDetector;
    private Button            btnBlur;
    private Button            btnExport;
    private ProgressBar       progressBar;
    private TextView          tvError;

    private final ActivityResultLauncher<String> pickLauncher =
        registerForActivityResult(new ActivityResultContracts.GetMultipleContents(), uris -> {
            if (uris != null && !uris.isEmpty()) {
                loadImages(uris);
            }
        });

    private final ActivityResultLauncher<Uri> folderLauncher =
        registerForActivityResult(new ActivityResultContracts.OpenDocumentTree(), treeUri -> {
            if (treeUri == null) return;
            // Persist permission so files remain accessible across restarts
            getContentResolver().takePersistableUriPermission(
                treeUri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION);
            loadImagesFromFolder(treeUri);
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_blur);

        ivMain        = findViewById(R.id.ivMain);
        btnToggle     = findViewById(R.id.btnToggle);
        tvRegions     = findViewById(R.id.tvRegions);
        tvCounter     = findViewById(R.id.tvCounter);
        navigationRow = findViewById(R.id.navigationRow);
        btnPrev       = findViewById(R.id.btnPrev);
        btnNext       = findViewById(R.id.btnNext);
        switchDetector= findViewById(R.id.switchDetector);
        btnBlur       = primaryButton = findViewById(R.id.btnBlur);
        btnExport     = findViewById(R.id.btnExport);
        progressBar   = findViewById(R.id.progressBar);
        tvError       = findViewById(R.id.tvError);

        findViewById(R.id.btnPick).setOnClickListener(v -> pickLauncher.launch("image/*"));
        findViewById(R.id.btnPickFolder).setOnClickListener(v -> folderLauncher.launch(null));
        btnBlur.setOnClickListener(v -> processCurrentImage());
        btnExport.setOnClickListener(v -> exportCurrentBlurred());
        btnToggle.setOnClickListener(v -> toggleView());
        btnPrev.setOnClickListener(v -> navigate(-1));
        btnNext.setOnClickListener(v -> navigate(+1));
    }

    // ── Image loading ─────────────────────────────────────────────────────────

    private void loadImages(List<Uri> uris) {
        imageUris.clear();
        origBoxed.clear();
        blurred.clear();
        imageUris.addAll(uris);
        currentIndex = 0;
        showBlurred  = false;
        onSelectionChanged();
    }

    private void loadImagesFromFolder(Uri treeUri) {
        // Run on background — DocumentFile traversal can be slow for large folders
        executor.execute(() -> {
            DocumentFile dir = DocumentFile.fromTreeUri(this, treeUri);
            if (dir == null || !dir.isDirectory()) {
                runOnUiThread(() -> showError("Directorul nu poate fi accesat"));
                return;
            }
            List<Uri> found = new ArrayList<>();
            collectImages(dir, found);
            if (found.isEmpty()) {
                runOnUiThread(() -> showError("Nicio imagine gasita in folder"));
                return;
            }
            runOnUiThread(() -> loadImages(found));
        });
    }

    /** Recursively collect image URIs from a DocumentFile directory. */
    private static void collectImages(DocumentFile dir, List<Uri> out) {
        DocumentFile[] children = dir.listFiles();
        if (children == null) return;
        for (DocumentFile f : children) {
            if (f.isDirectory()) {
                collectImages(f, out); // recurse into subdirectories
            } else {
                String mime = f.getType();
                if (mime != null && mime.startsWith("image/")) {
                    out.add(f.getUri());
                }
            }
        }
    }

    // ── Selection ────────────────────────────────────────────────────────────

    private void onSelectionChanged() {
        int n = imageUris.size();
        tvCounter.setVisibility(n > 1 ? View.VISIBLE : View.GONE);
        navigationRow.setVisibility(n > 1 ? View.VISIBLE : View.GONE);
        btnBlur.setEnabled(n > 0);
        btnExport.setEnabled(false);
        btnToggle.setVisibility(View.GONE);
        tvRegions.setVisibility(View.GONE);
        tvError.setVisibility(View.GONE);
        showCurrentOriginal();
        updateCounter();
    }

    private void navigate(int delta) {
        int n = imageUris.size();
        if (n == 0) return;
        currentIndex = (currentIndex + delta + n) % n;
        showBlurred = blurred.containsKey(currentIndex);
        showCurrentImage();
        updateCounter();
        btnExport.setEnabled(blurred.containsKey(currentIndex));
        boolean hasResults = blurred.containsKey(currentIndex);
        btnToggle.setVisibility(hasResults ? View.VISIBLE : View.GONE);
        tvRegions.setVisibility(hasResults && !showBlurred ? View.VISIBLE : View.GONE);
    }

    private void updateCounter() {
        int n = imageUris.size();
        tvCounter.setText((currentIndex + 1) + " / " + n);
    }

    // ── Image display ─────────────────────────────────────────────────────────

    private void showCurrentOriginal() {
        executor.execute(() -> {
            try {
                Bitmap bmp = ImageUtils.decodeBitmap(getContentResolver(), imageUris.get(currentIndex), 1200);
                runOnUiThread(() -> { ivMain.resetZoom(); ivMain.setImageBitmap(bmp); });
            } catch (IOException e) {
                runOnUiThread(() -> showError("Nu s-a putut incarca imaginea"));
            }
        });
    }

    private void showCurrentImage() {
        ivMain.resetZoom();
        if (showBlurred && blurred.containsKey(currentIndex)) {
            ivMain.setImageBitmap(blurred.get(currentIndex));
            btnToggle.setText("BLURRED");
        } else if (origBoxed.containsKey(currentIndex)) {
            ivMain.setImageBitmap(origBoxed.get(currentIndex));
            btnToggle.setText("ORIGINAL");
        } else {
            showCurrentOriginal();
        }
    }

    private void toggleView() {
        showBlurred = !showBlurred;
        showCurrentImage();
        tvRegions.setVisibility(!showBlurred && origBoxed.containsKey(currentIndex)
            ? View.VISIBLE : View.GONE);
    }

    // ── Processing ────────────────────────────────────────────────────────────

    private void processCurrentImage() {
        if (imageUris.isEmpty()) return;
        setProcessing(true);
        btnBlur.setEnabled(false);
        btnExport.setEnabled(false);
        btnToggle.setVisibility(View.GONE);
        tvError.setVisibility(View.GONE);
        progressBar.setIndeterminate(true);

        String detector = switchDetector.isChecked() ? "gemini" : "local";
        int idx = currentIndex;
        Uri uri = imageUris.get(idx);

        executor.execute(() -> {
            try {
                byte[] imageBytes = readBytes(uri);
                if (imageBytes == null) throw new IOException("Nu s-a putut citi imaginea");

                // 1. Detect regions (for bounding boxes on original)
                MultipartBody.Part detectPart = buildPart("file", imageBytes);
                Response<BlurDetectResponse> detectResp =
                    api().blurDetect(detectPart, detector).execute();

                List<BlurRegion> regions = new ArrayList<>();
                if (detectResp.isSuccessful() && detectResp.body() != null) {
                    regions = detectResp.body().regions;
                }

                // 2. Draw boxes on original bitmap
                Bitmap original = ImageUtils.decodeBitmap(getContentResolver(), uri, 1200);
                Bitmap withBoxes = drawBoxes(original, regions);
                if (withBoxes != original) original.recycle();

                // 3. Get blurred result
                MultipartBody.Part blurPart = buildPart("file", imageBytes);
                Response<ResponseBody> blurResp =
                    api().blurSensitive(blurPart, detector).execute();

                if (!blurResp.isSuccessful() || blurResp.body() == null)
                    throw new IOException("Blur esuat: HTTP " + blurResp.code());

                byte[] blurBytes = blurResp.body().bytes();
                Bitmap blurredBmp = BitmapFactory.decodeByteArray(blurBytes, 0, blurBytes.length);
                if (blurredBmp == null) throw new IOException("Rezultat blur invalid");

                // Store results
                origBoxed.put(idx, withBoxes);
                blurred.put(idx, blurredBmp);
                final int regionCount = regions.size();
                showBlurred = true;

                runOnUiThread(() -> {
                    setProcessing(false);
                    btnBlur.setEnabled(true);
                    btnExport.setEnabled(true);
                    btnToggle.setVisibility(View.VISIBLE);
                    showCurrentImage();
                    if (regionCount > 0) {
                        tvRegions.setText(regionCount + " zone detectate");
                    } else {
                        tvRegions.setText("0 zone detectate");
                    }
                });

            } catch (Exception e) {
                runOnUiThread(() -> {
                    setProcessing(false);
                    btnBlur.setEnabled(true);
                    showError("Eroare: " + e.getMessage());
                });
            }
        });
    }

    // ── Export ────────────────────────────────────────────────────────────────

    private void exportCurrentBlurred() {
        Bitmap bmp = blurred.get(currentIndex);
        if (bmp == null) return;

        executor.execute(() -> {
            try {
                String filename = "blur_" + System.currentTimeMillis() + ".jpg";
                ContentValues cv = new ContentValues();
                cv.put(MediaStore.Images.Media.DISPLAY_NAME, filename);
                cv.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    cv.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/PhotoMatch");
                } else {
                    File dir = new File(Environment.getExternalStoragePublicDirectory(
                        Environment.DIRECTORY_PICTURES), "PhotoMatch");
                    dir.mkdirs();
                    cv.put(MediaStore.Images.Media.DATA, new File(dir, filename).getAbsolutePath());
                }
                Uri insertUri = getContentResolver().insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv);
                if (insertUri == null) throw new IOException("MediaStore insert null");
                try (OutputStream os = getContentResolver().openOutputStream(insertUri)) {
                    bmp.compress(Bitmap.CompressFormat.JPEG, 95, os);
                }
                runOnUiThread(() -> Toast.makeText(this, "Salvat in galerie", Toast.LENGTH_SHORT).show());
            } catch (IOException e) {
                runOnUiThread(() -> showError("Eroare salvare: " + e.getMessage()));
            }
        });
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private byte[] readBytes(Uri uri) {
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            return is == null ? null : is.readAllBytes();
        } catch (IOException e) {
            return null;
        }
    }

    private static MultipartBody.Part buildPart(String field, byte[] bytes) {
        RequestBody body = RequestBody.create(bytes, MediaType.parse("image/jpeg"));
        return MultipartBody.Part.createFormData(field, "photo.jpg", body);
    }

    private static Bitmap drawBoxes(Bitmap src, List<BlurRegion> regions) {
        if (regions == null || regions.isEmpty()) return src;
        Bitmap result = src.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(result);

        Paint boxPaint = new Paint();
        boxPaint.setColor(Color.parseColor("#C9A84C")); // amber
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(Math.max(3f, result.getWidth() / 300f));
        boxPaint.setAntiAlias(true);

        Paint labelBg = new Paint();
        labelBg.setColor(Color.parseColor("#CC000000"));

        Paint labelPaint = new Paint();
        labelPaint.setColor(Color.parseColor("#C9A84C"));
        labelPaint.setTextSize(Math.max(24f, result.getWidth() / 40f));
        labelPaint.setAntiAlias(true);
        labelPaint.setTypeface(android.graphics.Typeface.MONOSPACE);

        for (BlurRegion r : regions) {
            canvas.drawRect(r.x, r.y, r.x + r.w, r.y + r.h, boxPaint);
            if (r.label != null && !r.label.isEmpty()) {
                float textH = labelPaint.getTextSize();
                canvas.drawRect(r.x, r.y, r.x + r.w, r.y + textH + 8, labelBg);
                canvas.drawText(r.label, r.x + 6, r.y + textH, labelPaint);
            }
        }
        return result;
    }

    @Override
    protected void showError(String msg) {
        tvError.setText(msg);
        tvError.setVisibility(View.VISIBLE);
    }
}
