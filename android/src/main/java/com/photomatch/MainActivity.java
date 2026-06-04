package com.photomatch;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.widget.Button;
import android.widget.Switch;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.FileProvider;

import com.photomatch.api.ApiClient;
import com.photomatch.api.UserProfile;
import com.photomatch.base.BaseActivity;
import com.photomatch.util.ImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.Executors;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends BaseActivity {

    public static final String EXTRA_IMAGE_PATH = "image_path";
    private static final String KEY_SESSION = "style_session_id";

    private File   cameraFile;
    private Switch switchStyle;

    private final ActivityResultLauncher<Uri> takePictureLauncher =
        registerForActivityResult(new ActivityResultContracts.TakePicture(), success -> {
            if (success && cameraFile != null && cameraFile.exists())
                compressAndLaunch(cameraFile);
        });

    private final ActivityResultLauncher<String> getContentLauncher =
        registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
            if (uri != null) compressAndLaunchUri(uri);
        });

    private final ActivityResultLauncher<String[]> permissionLauncher =
        registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), r -> {});

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Server IP prompt on first launch
        SharedPreferences prefs = getSharedPreferences(ApiClient.PREFS_NAME, MODE_PRIVATE);
        if (!prefs.contains(ApiClient.KEY_SERVER_IP)) showServerIpDialog();

        // Profile strip
        TextView tvEmail = findViewById(R.id.tvUserEmail);
        api().getMe().enqueue(new Callback<UserProfile>() {
            @Override public void onResponse(Call<UserProfile> c, Response<UserProfile> r) {
                if (r.isSuccessful() && r.body() != null)
                    tvEmail.setText(r.body().email);
            }
            @Override public void onFailure(Call<UserProfile> c, Throwable t) { /* keeps "···" */ }
        });
        findViewById(R.id.btnLogout).setOnClickListener(v -> logout());

        // Style toggle
        switchStyle = findViewById(R.id.switchStyle);
        switchStyle.setChecked(prefs.getBoolean("style_enabled", false));
        switchStyle.setOnCheckedChangeListener((b, checked) ->
            prefs.edit().putBoolean("style_enabled", checked).apply());
        findViewById(R.id.tvSetupStyle).setOnClickListener(v ->
            startActivity(new Intent(this, StyleSetupActivity.class)));

        // EDITARE
        Button btnCamera = findViewById(R.id.btnCamera);
        btnCamera.setOnClickListener(v -> {
            if (!hasPermission(Manifest.permission.CAMERA)) {
                permissionLauncher.launch(new String[]{ Manifest.permission.CAMERA });
                showError("Permisiune necesara — apasa din nou dupa acordare");
                return;
            }
            launchCamera();
        });

        Button btnGallery = findViewById(R.id.btnGallery);
        btnGallery.setOnClickListener(v -> {
            String perm = Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                ? Manifest.permission.READ_MEDIA_IMAGES
                : Manifest.permission.READ_EXTERNAL_STORAGE;
            if (!hasPermission(perm)) {
                permissionLauncher.launch(new String[]{ perm });
                showError("Permisiune necesara — apasa din nou dupa acordare");
                return;
            }
            getContentLauncher.launch("image/*");
        });

        findViewById(R.id.btnBatch).setOnClickListener(v ->
            startActivity(new Intent(this, BatchActivity.class)));
        findViewById(R.id.btnPipeline).setOnClickListener(v ->
            startActivity(new Intent(this, PipelineActivity.class)));

        // ORGANIZARE
        findViewById(R.id.btnBurst).setOnClickListener(v ->
            startActivity(new Intent(this, BurstActivity.class)));
        findViewById(R.id.btnCluster).setOnClickListener(v ->
            startActivity(new Intent(this, ClusterActivity.class)));
        findViewById(R.id.btnFaces).setOnClickListener(v ->
            startActivity(new Intent(this, FaceGroupsActivity.class)));

        // DISTRIBUIRE
        findViewById(R.id.btnDelivery).setOnClickListener(v ->
            startActivity(new Intent(this, DeliveryActivity.class)));

        // PROTECTIE
        findViewById(R.id.btnBlur).setOnClickListener(v ->
            startActivity(new Intent(this, BlurActivity.class)));
        findViewById(R.id.btnWebScraper).setOnClickListener(v ->
            startActivity(new Intent(this, WebScraperActivity.class)));

        // BIBLIOTECA
        findViewById(R.id.btnFavorites).setOnClickListener(v ->
            startActivity(new Intent(this, FavoritesActivity.class)));

        // SERVER IP
        findViewById(R.id.btnServerIp).setOnClickListener(v -> showServerIpDialog());
    }

    // ── Camera + Gallery ─────────────────────────────────────────────────────

    private void launchCamera() {
        try {
            cameraFile = File.createTempFile("capture_", ".jpg", getExternalCacheDir());
            Uri uri = FileProvider.getUriForFile(this, "com.photomatch.fileprovider", cameraFile);
            takePictureLauncher.launch(uri);
        } catch (IOException e) {
            showError("Camera error: " + e.getMessage());
        }
    }

    private void compressAndLaunch(File source) {
        Executors.newSingleThreadExecutor().execute(() -> {
            try {
                Bitmap bmp = ImageUtils.decodeBitmap(source.getAbsolutePath(), 1200);
                File out = new File(getCacheDir(), "to_process.jpg");
                try (FileOutputStream fos = new FileOutputStream(out)) {
                    bmp.compress(Bitmap.CompressFormat.JPEG, 85, fos);
                }
                bmp.recycle();
                startProcessing(out.getAbsolutePath());
            } catch (IOException e) {
                runOnUiThread(() -> showError("Failed to process image"));
            }
        });
    }

    private void compressAndLaunchUri(Uri uri) {
        Executors.newSingleThreadExecutor().execute(() -> {
            try {
                Bitmap bmp = ImageUtils.decodeBitmap(getContentResolver(), uri, 1200);
                File out = new File(getCacheDir(), "to_process.jpg");
                try (FileOutputStream fos = new FileOutputStream(out)) {
                    bmp.compress(Bitmap.CompressFormat.JPEG, 85, fos);
                }
                bmp.recycle();
                startProcessing(out.getAbsolutePath());
            } catch (IOException e) {
                runOnUiThread(() -> showError("Failed to load image"));
            }
        });
    }

    private void startProcessing(String imagePath) {
        boolean useStyle = switchStyle.isChecked();
        String sessionId = useStyle
            ? getSharedPreferences(ApiClient.PREFS_NAME, MODE_PRIVATE).getString(KEY_SESSION, null)
            : null;

        if (useStyle && sessionId == null) {
            runOnUiThread(() -> {
                showError("Seteaza-ti stilul mai intai");
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

    private boolean hasPermission(String permission) {
        return checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED;
    }
}
