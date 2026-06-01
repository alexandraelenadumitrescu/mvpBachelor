package com.photomatch.lite.ui;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import java.util.List;
import com.photomatch.lite.R;
import com.photomatch.lite.face.FaceClusterer;
import com.photomatch.lite.face.FaceDetectorHelper;
import com.photomatch.lite.face.FaceEmbedder;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ClusterActivity extends AppCompatActivity {

    private TextView      tvCount;
    private TextView      tvSummary;
    private ProgressBar   progressBar;
    private LinearLayout  clustersContainer;
    private List<Uri>     selectedUris = new ArrayList<>();

    private ExecutorService    executor;
    private FaceDetectorHelper faceDetector;
    private FaceEmbedder       faceEmbedder;

    private final ActivityResultLauncher<String> pickPhotos =
        registerForActivityResult(new ActivityResultContracts.GetMultipleContents(), uris -> {
            selectedUris.addAll(uris);
            tvCount.setText(selectedUris.size() + " fotografii selectate");
        });

    private final ActivityResultLauncher<Uri> pickFolder =
        registerForActivityResult(new ActivityResultContracts.OpenDocumentTree(), uri -> {
            if (uri == null) return;
            List<Uri> images = FolderPickerHelper.listImages(this, uri);
            selectedUris.addAll(images);
            tvCount.setText(selectedUris.size() + " fotografii selectate");
            Toast.makeText(this, images.size() + " fotografii adăugate din folder",
                Toast.LENGTH_SHORT).show();
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_cluster);

        tvCount           = findViewById(R.id.tvCount);
        tvSummary         = findViewById(R.id.tvSummary);
        progressBar       = findViewById(R.id.progressBar);
        clustersContainer = findViewById(R.id.clustersContainer);

        executor     = Executors.newSingleThreadExecutor();
        faceDetector = new FaceDetectorHelper();
        try {
            faceEmbedder = new FaceEmbedder(this);
        } catch (RuntimeException e) {
            Toast.makeText(this, "facenet.tflite lipsă din assets", Toast.LENGTH_LONG).show();
        }

        findViewById(R.id.btnPick).setOnClickListener(v -> pickPhotos.launch("image/*"));
        findViewById(R.id.btnPickFolder).setOnClickListener(v -> pickFolder.launch(null));
        findViewById(R.id.btnClear).setOnClickListener(v -> {
            selectedUris.clear();
            tvCount.setText("Nicio fotografie selectată");
        });
        findViewById(R.id.btnCluster).setOnClickListener(v -> runClustering());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor    != null) executor.shutdownNow();
        if (faceDetector != null) faceDetector.close();
        if (faceEmbedder != null) faceEmbedder.close();
    }

    private void runClustering() {
        if (selectedUris.isEmpty()) {
            Toast.makeText(this, "Selectează fotografii mai întâi", Toast.LENGTH_SHORT).show();
            return;
        }
        if (faceEmbedder == null) {
            Toast.makeText(this, "Modelul nu a putut fi încărcat", Toast.LENGTH_SHORT).show();
            return;
        }

        progressBar.setVisibility(View.VISIBLE);
        clustersContainer.removeAllViews();
        tvSummary.setText("Procesare...");
        tvCount.setEnabled(false);
        findViewById(R.id.btnPick).setEnabled(false);
        findViewById(R.id.btnCluster).setEnabled(false);

        executor.execute(() -> {
            List<FaceClusterer.FaceEmbedding> allFaces = new ArrayList<>();

            for (int i = 0; i < selectedUris.size(); i++) {
                final int idx = i;
                final int total = selectedUris.size();
                runOnUiThread(() -> tvSummary.setText(
                    "Poză " + (idx + 1) + " / " + total + "..."));

                Uri uri = selectedUris.get(i);
                Bitmap bmp = decodeBitmap(uri, 1024);
                if (bmp == null) continue;

                try {
                    List<Rect> boxes = faceDetector.detect(bmp);
                    for (int f = 0; f < boxes.size(); f++) {
                        Bitmap crop  = cropFace(bmp, boxes.get(f));
                        float[] emb  = faceEmbedder.embed(crop);
                        Bitmap thumb = Bitmap.createScaledBitmap(crop, 112, 112, true);
                        if (thumb != crop) crop.recycle();
                        allFaces.add(new FaceClusterer.FaceEmbedding(
                            uri.toString(), thumb, emb, f));
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                bmp.recycle();
            }

            List<FaceClusterer.FaceCluster> clusters = FaceClusterer.cluster(allFaces);
            runOnUiThread(() -> showResults(clusters));
        });
    }

    private void showResults(List<FaceClusterer.FaceCluster> clusters) {
        progressBar.setVisibility(View.GONE);
        tvCount.setEnabled(true);
        findViewById(R.id.btnPick).setEnabled(true);
        findViewById(R.id.btnCluster).setEnabled(true);

        long persons  = clusters.stream().filter(c -> c.personIndex > 0).count();
        long singles  = clusters.stream().filter(c -> c.personIndex == 0).count();
        int  total    = clusters.stream().mapToInt(c -> c.faces.size()).sum();
        tvSummary.setText(String.format(Locale.US,
            "%d fețe detectate · %d persoane · %d unice", total, persons, singles));

        for (FaceClusterer.FaceCluster cluster : clusters) {
            TextView title = new TextView(this);
            if (cluster.personIndex > 0) {
                title.setText(String.format(Locale.US,
                    "Persoana %d  ·  %d apariții", cluster.personIndex, cluster.faces.size()));
            } else {
                title.setText("Față unică");
            }
            title.setTextSize(15);
            title.setPadding(0, 20, 0, 6);
            clustersContainer.addView(title);

            HorizontalScrollView hscroll = new HorizontalScrollView(this);
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);

            for (FaceClusterer.FaceEmbedding face : cluster.faces) {
                ImageView iv = new ImageView(this);
                int size = (int) (100 * getResources().getDisplayMetrics().density);
                LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(size, size);
                lp.setMargins(4, 4, 4, 4);
                iv.setLayoutParams(lp);
                iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
                iv.setImageBitmap(face.crop);
                row.addView(iv);
            }

            hscroll.addView(row);
            clustersContainer.addView(hscroll);

            View divider = new View(this);
            divider.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 1));
            divider.setBackgroundColor(0xFFDDDDDD);
            clustersContainer.addView(divider);
        }
    }

    private Bitmap decodeBitmap(Uri uri, int maxSide) {
        try {
            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            try (InputStream is = getContentResolver().openInputStream(uri)) {
                BitmapFactory.decodeStream(is, null, opts);
            }
            int sample = 1;
            while (opts.outWidth / sample > maxSide || opts.outHeight / sample > maxSide) sample *= 2;
            opts.inSampleSize = sample;
            opts.inJustDecodeBounds = false;
            try (InputStream is = getContentResolver().openInputStream(uri)) {
                return BitmapFactory.decodeStream(is, null, opts);
            }
        } catch (IOException e) {
            return null;
        }
    }

    private Bitmap cropFace(Bitmap src, Rect box) {
        int pad = (int) (Math.min(box.width(), box.height()) * 0.20f);
        int l = Math.max(0, box.left   - pad);
        int t = Math.max(0, box.top    - pad);
        int r = Math.min(src.getWidth(),  box.right  + pad);
        int b = Math.min(src.getHeight(), box.bottom + pad);
        return Bitmap.createBitmap(src, l, t, r - l, b - t);
    }
}
