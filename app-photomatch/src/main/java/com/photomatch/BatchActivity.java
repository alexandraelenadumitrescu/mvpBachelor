package com.photomatch;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.gson.Gson;
import com.photomatch.api.ApiClient;
import com.photomatch.api.BatchResponse;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
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

public class BatchActivity extends AppCompatActivity {

    private static final int MAX_PHOTOS = 100;
    static final String EXTRA_CACHE_PATH = "cache_path";

    private List<Uri> selectedUris = new ArrayList<>();

    private Button      btnPick;
    private Button      btnProcess;
    private TextView    tvCount;
    private TextView    tvProgress;
    private TextView    tvError;
    private ProgressBar progressBar;
    private RecyclerView rvThumbnails;
    private ThumbnailAdapter thumbnailAdapter;

    private ExecutorService executor;

    private final ActivityResultLauncher<String> pickLauncher =
        registerForActivityResult(new ActivityResultContracts.GetMultipleContents(), uris -> {
            if (uris != null && !uris.isEmpty()) {
                selectedUris = new ArrayList<>(uris.subList(0, Math.min(uris.size(), MAX_PHOTOS)));
                updateSelectionUI();
            }
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_batch);

        btnPick      = findViewById(R.id.btnPick);
        btnProcess   = findViewById(R.id.btnProcess);
        tvCount      = findViewById(R.id.tvCount);
        tvProgress   = findViewById(R.id.tvProgress);
        tvError      = findViewById(R.id.tvError);
        progressBar  = findViewById(R.id.progressBar);
        rvThumbnails = findViewById(R.id.rvThumbnails);

        thumbnailAdapter = new ThumbnailAdapter();
        rvThumbnails.setLayoutManager(
            new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false));
        rvThumbnails.setAdapter(thumbnailAdapter);

        btnPick.setOnClickListener(v -> pickLauncher.launch("image/*"));
        btnProcess.setOnClickListener(v -> startBatchProcessing());

        executor = Executors.newSingleThreadExecutor();

        updateProcessButton();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor != null) executor.shutdownNow();
    }

    private void updateSelectionUI() {
        int n = selectedUris.size();
        tvCount.setText(n + " photo" + (n == 1 ? "" : "s") + " selected");
        thumbnailAdapter.setUris(selectedUris);
        updateProcessButton();
    }

    private void updateProcessButton() {
        int n = selectedUris.size();
        if (n == 0) {
            btnProcess.setText("PROCESS 0 PHOTOS");
            btnProcess.setEnabled(false);
        } else {
            btnProcess.setText("PROCESS " + n + " PHOTO" + (n == 1 ? "" : "S"));
            btnProcess.setEnabled(true);
        }
    }

    private void showError(String msg) {
        tvError.setText(msg);
        tvError.setVisibility(View.VISIBLE);
    }

    private void setUiProcessing(boolean processing) {
        btnPick.setEnabled(!processing);
        btnProcess.setEnabled(!processing);
        tvProgress.setVisibility(processing ? View.VISIBLE : View.GONE);
        progressBar.setVisibility(processing ? View.VISIBLE : View.GONE);
        tvError.setVisibility(View.GONE);
    }

    private void startBatchProcessing() {
        if (selectedUris.isEmpty()) return;
        setUiProcessing(true);
        progressBar.setMax(selectedUris.size());

        executor.execute(() -> {
            try {
                // Phase 1: compress all images on-device + run blur detection
                List<MultipartBody.Part> parts     = new ArrayList<>();
                List<Boolean>           blurFlags  = new ArrayList<>();
                for (int i = 0; i < selectedUris.size(); i++) {
                    final int idx = i;
                    runOnUiThread(() -> {
                        tvProgress.setText("Compressing " + (idx + 1) + "/" + selectedUris.size() + "...");
                        progressBar.setProgress(idx + 1);
                    });
                    File tmp = compressUri(selectedUris.get(i), i);
                    RequestBody rb = RequestBody.create(MediaType.parse("image/jpeg"), tmp);
                    parts.add(MultipartBody.Part.createFormData("files", tmp.getName(), rb));

                    Bitmap bmp = BitmapFactory.decodeFile(tmp.getAbsolutePath());
                    if (bmp != null) {
                        blurFlags.add(BlurDetector.check(bmp).isBlurry);
                        bmp.recycle();
                    } else {
                        blurFlags.add(false);
                    }
                }

                // Phase 2: single server request
                runOnUiThread(() -> {
                    tvProgress.setText("Processing on server...");
                    progressBar.setIndeterminate(true);
                });

                Response<BatchResponse> resp = ApiClient.getInstance()
                    .getService().batchProcess(parts).execute();

                if (!resp.isSuccessful() || resp.body() == null) {
                    String err = resp.errorBody() != null ? resp.errorBody().string() : "";
                    throw new IOException("Server error HTTP " + resp.code() + " " + err);
                }

                // Cache results and navigate
                BatchCache cache = new BatchCache();
                cache.response     = resp.body();
                cache.originalUris = urisToStrings(selectedUris);
                cache.blurFlags    = blurFlags;
                String json = new Gson().toJson(cache);

                File outFile = new File(getExternalFilesDir(null),
                    "batch_" + System.currentTimeMillis() + ".json");
                try (FileWriter fw = new FileWriter(outFile)) {
                    fw.write(json);
                }

                Intent intent = new Intent(this, BatchResultsActivity.class);
                intent.putExtra(EXTRA_CACHE_PATH, outFile.getAbsolutePath());
                runOnUiThread(() -> {
                    startActivity(intent);
                    finish();
                });

            } catch (Exception e) {
                runOnUiThread(() -> {
                    setUiProcessing(false);
                    showError("Error: " + e.getMessage());
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
        File out = new File(getCacheDir(), "batch_" + index + ".jpg");
        try (FileOutputStream fos = new FileOutputStream(out)) {
            bmp.compress(Bitmap.CompressFormat.JPEG, 85, fos);
        }
        bmp.recycle();
        return out;
    }

    private static List<String> urisToStrings(List<Uri> uris) {
        List<String> out = new ArrayList<>(uris.size());
        for (Uri u : uris) out.add(u.toString());
        return out;
    }

    // --- Thumbnail strip adapter ---

    private class ThumbnailAdapter extends RecyclerView.Adapter<ThumbnailAdapter.VH> {

        private List<Uri> uris = new ArrayList<>();

        void setUris(List<Uri> uris) {
            this.uris = uris;
            notifyDataSetChanged();
        }

        @Override
        public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            ImageView iv = new ImageView(parent.getContext());
            int sizePx = (int) (80 * getResources().getDisplayMetrics().density);
            int marginPx = (int) (2 * getResources().getDisplayMetrics().density);
            RecyclerView.LayoutParams lp = new RecyclerView.LayoutParams(sizePx, sizePx);
            lp.setMargins(marginPx, marginPx, marginPx, marginPx);
            iv.setLayoutParams(lp);
            iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
            return new VH(iv);
        }

        @Override
        public void onBindViewHolder(VH holder, int position) {
            Uri uri = uris.get(position);
            Executors.newSingleThreadExecutor().execute(() -> {
                try {
                    BitmapFactory.Options opts = new BitmapFactory.Options();
                    opts.inSampleSize = 4;
                    opts.inJustDecodeBounds = false;
                    Bitmap bmp;
                    try (InputStream is = getContentResolver().openInputStream(uri)) {
                        bmp = BitmapFactory.decodeStream(is, null, opts);
                    }
                    if (bmp != null) runOnUiThread(() -> holder.iv.setImageBitmap(bmp));
                } catch (IOException ignored) {}
            });
        }

        @Override public int getItemCount() { return uris.size(); }

        class VH extends RecyclerView.ViewHolder {
            final ImageView iv;
            VH(ImageView iv) { super(iv); this.iv = iv; }
        }
    }

    // --- Cache POJO ---

    public static class BatchCache {
        public BatchResponse  response;
        public List<String>   originalUris;
        public List<Boolean>  blurFlags;
    }
}
