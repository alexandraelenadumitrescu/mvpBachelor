package com.photomatch;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import com.photomatch.api.ApiClient;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    public static final String EXTRA_IMAGE_PATH = "image_path";

    private static final String PREFS       = "photomatch_prefs";
    private static final String KEY_SESSION = "style_session_id";

    private File   cameraFile;
    private Switch switchStyle;

    // --- Activity result launchers ---

    private final ActivityResultLauncher<Uri> takePictureLauncher =
        registerForActivityResult(new ActivityResultContracts.TakePicture(), success -> {
            if (success && cameraFile != null && cameraFile.exists()) {
                compressAndLaunch(cameraFile);
            }
        });

    private final ActivityResultLauncher<String> getContentLauncher =
        registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
            if (uri != null) {
                compressAndLaunchUri(uri);
            }
        });

    private final ActivityResultLauncher<String[]> permissionLauncher =
        registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), results -> {
            // After permissions answered, the user will tap the button again.
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ApiClient.init(this);
        setContentView(R.layout.activity_main);

        SharedPreferences prefs = getSharedPreferences(ApiClient.PREFS_NAME, MODE_PRIVATE);
        if (!prefs.contains(ApiClient.KEY_SERVER_IP)) {
            showServerIpDialog();
        }

        Button   btnCamera  = findViewById(R.id.btnCamera);
        Button   btnGallery = findViewById(R.id.btnGallery);
        Button   btnBatch   = findViewById(R.id.btnBatch);
        switchStyle         = findViewById(R.id.switchStyle);
        TextView tvSetup    = findViewById(R.id.tvSetupStyle);

        // Restore switch state from prefs
        switchStyle.setChecked(prefs.getBoolean("style_enabled", false));
        switchStyle.setOnCheckedChangeListener((btn, checked) ->
            prefs.edit().putBoolean("style_enabled", checked).apply());

        tvSetup.setOnClickListener(v ->
            startActivity(new Intent(this, StyleSetupActivity.class)));

        btnBatch.setOnClickListener(v ->
            startActivity(new Intent(this, BatchActivity.class)));

        Button btnCluster = findViewById(R.id.btnCluster);
        btnCluster.setOnClickListener(v ->
            startActivity(new Intent(this, ClusterActivity.class)));

        Button btnBurst = findViewById(R.id.btnBurst);
        btnBurst.setOnClickListener(v ->
            startActivity(new Intent(this, BurstActivity.class)));

        Button btnFavorites = findViewById(R.id.btnFavorites);
        btnFavorites.setOnClickListener(v ->
            startActivity(new Intent(this, FavoritesActivity.class)));

        Button btnFaces = findViewById(R.id.btnFaces);
        btnFaces.setOnClickListener(v ->
            startActivity(new Intent(this, FaceGroupsActivity.class)));

        Button btnPipeline = findViewById(R.id.btnPipeline);
        btnPipeline.setOnClickListener(v ->
            startActivity(new Intent(this, PipelineActivity.class)));

        Button btnServerIp = findViewById(R.id.btnServerIp);
        btnServerIp.setOnClickListener(v -> showServerIpDialog());

        btnCamera.setOnClickListener(v -> {
            if (!hasPermission(Manifest.permission.CAMERA)) {
                requestPermissions(new String[]{ Manifest.permission.CAMERA });
                return;
            }
            launchCamera();
        });

        btnGallery.setOnClickListener(v -> {
            String storagePermission = Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                ? Manifest.permission.READ_MEDIA_IMAGES
                : Manifest.permission.READ_EXTERNAL_STORAGE;
            if (!hasPermission(storagePermission)) {
                requestPermissions(new String[]{ storagePermission });
                return;
            }
            getContentLauncher.launch("image/*");
        });
    }

    private void launchCamera() {
        try {
            cameraFile = File.createTempFile("capture_", ".jpg", getExternalCacheDir());
            Uri cameraUri = FileProvider.getUriForFile(
                this, "com.photomatch.fileprovider", cameraFile);
            takePictureLauncher.launch(cameraUri);
        } catch (IOException e) {
            Toast.makeText(this, "Camera error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    /** Compress a File (from camera) and navigate to ProcessingActivity. */
    private void compressAndLaunch(File sourceFile) {
        Executors.newSingleThreadExecutor().execute(() -> {
            try {
                File out = compress(sourceFile);
                startProcessing(out.getAbsolutePath());
            } catch (IOException e) {
                runOnUiThread(() ->
                    Toast.makeText(this, "Failed to process image", Toast.LENGTH_SHORT).show());
            }
        });
    }

    /** Compress a content:// URI (from gallery) and navigate to ProcessingActivity. */
    private void compressAndLaunchUri(Uri uri) {
        Executors.newSingleThreadExecutor().execute(() -> {
            try {
                BitmapFactory.Options opts = new BitmapFactory.Options();
                opts.inJustDecodeBounds = true;
                try (InputStream is = getContentResolver().openInputStream(uri)) {
                    BitmapFactory.decodeStream(is, null, opts);
                }
                opts.inSampleSize = computeSampleSize(opts.outWidth, opts.outHeight, 1200);
                opts.inJustDecodeBounds = false;

                Bitmap bmp;
                try (InputStream is = getContentResolver().openInputStream(uri)) {
                    bmp = BitmapFactory.decodeStream(is, null, opts);
                }
                if (bmp == null) throw new IOException("Could not decode image");

                File out = new File(getCacheDir(), "to_process.jpg");
                try (FileOutputStream fos = new FileOutputStream(out)) {
                    bmp.compress(Bitmap.CompressFormat.JPEG, 85, fos);
                }
                bmp.recycle();

                startProcessing(out.getAbsolutePath());
            } catch (IOException e) {
                runOnUiThread(() ->
                    Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show());
            }
        });
    }

    /** Compress a File (from camera), sampling down to max 1200px on the long side. */
    private File compress(File source) throws IOException {
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(source.getAbsolutePath(), opts);
        opts.inSampleSize = computeSampleSize(opts.outWidth, opts.outHeight, 1200);
        opts.inJustDecodeBounds = false;

        Bitmap bmp = BitmapFactory.decodeFile(source.getAbsolutePath(), opts);
        if (bmp == null) throw new IOException("Could not decode camera image");

        File out = new File(getCacheDir(), "to_process.jpg");
        try (FileOutputStream fos = new FileOutputStream(out)) {
            bmp.compress(Bitmap.CompressFormat.JPEG, 85, fos);
        }
        bmp.recycle();
        return out;
    }

    private void startProcessing(String imagePath) {
        boolean useStyle = switchStyle.isChecked();
        String sessionId = useStyle
            ? getSharedPreferences(PREFS, MODE_PRIVATE).getString(KEY_SESSION, null)
            : null;

        if (useStyle && sessionId == null) {
            runOnUiThread(() -> {
                Toast.makeText(this, "Set up your style first", Toast.LENGTH_SHORT).show();
                startActivity(new Intent(this, StyleSetupActivity.class));
            });
            return;
        }

        Intent intent = new Intent(this, ProcessingActivity.class);
        intent.putExtra(EXTRA_IMAGE_PATH, imagePath);
        intent.putExtra("use_style", useStyle && sessionId != null);
        intent.putExtra("session_id", sessionId);
        runOnUiThread(() -> startActivity(intent));
    }

    private void showServerIpDialog() {
        EditText input = new EditText(this);
        input.setHint("ex: 10.33.128.137");
        input.setText(ApiClient.getServerIp());
        input.setInputType(android.text.InputType.TYPE_CLASS_TEXT);

        new AlertDialog.Builder(this)
            .setTitle("Server IP")
            .setMessage("Introdu IP-ul laptopului (din ipconfig)")
            .setView(input)
            .setPositiveButton("Salveaza", (dialog, which) -> {
                String ip = input.getText().toString().trim();
                if (!ip.isEmpty()) {
                    ApiClient.saveServerIp(ip);
                    Toast.makeText(this, "IP salvat: " + ip, Toast.LENGTH_SHORT).show();
                }
            })
            .setNegativeButton("Anuleaza", null)
            .show();
    }

    private boolean hasPermission(String permission) {
        return checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermissions(String[] permissions) {
        permissionLauncher.launch(permissions);
        Toast.makeText(this, "Permission required — please tap the button again after granting.",
            Toast.LENGTH_LONG).show();
    }

    static int computeSampleSize(int width, int height, int maxSide) {
        int inSampleSize = 1;
        int maxDim = Math.max(width, height);
        while (maxDim / (inSampleSize * 2) > maxSide) {
            inSampleSize *= 2;
        }
        return inSampleSize;
    }
}
