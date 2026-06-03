package com.photomatch;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class FaceGroupsActivity extends AppCompatActivity {

    private static final int   MAX_PHOTOS          = 50;
    private static final String PREFS_NAME         = "face_prefs";
    private static final String KEY_PRIVACY_SHOWN  = "face_privacy_accepted";

    private List<Uri> selectedUris = new ArrayList<>();

    private LinearLayout selectionSection;
    private LinearLayout resultsSection;
    private RecyclerView rvThumbnails;
    private TextView     tvCount;
    private Button       btnPick;
    private Button       btnAnalyze;
    private ProgressBar  progressBar;
    private TextView     tvProgress;
    private TextView     tvError;
    private RecyclerView rvGroups;
    private TextView     tvResultsHeader;

    private ExecutorService    executor;
    private FaceDetectorHelper faceDetector;
    private FaceEmbedder       faceEmbedder;

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
        setContentView(R.layout.activity_face_groups);

        selectionSection = findViewById(R.id.selectionSection);
        resultsSection   = findViewById(R.id.resultsSection);
        rvThumbnails     = findViewById(R.id.rvThumbnails);
        tvCount          = findViewById(R.id.tvCount);
        btnPick          = findViewById(R.id.btnPick);
        btnAnalyze       = findViewById(R.id.btnAnalyze);
        progressBar      = findViewById(R.id.progressBar);
        tvProgress       = findViewById(R.id.tvProgress);
        tvError          = findViewById(R.id.tvError);
        rvGroups         = findViewById(R.id.rvGroups);
        tvResultsHeader  = findViewById(R.id.tvResultsHeader);

        ThumbnailAdapter thumbAdapter = new ThumbnailAdapter();
        rvThumbnails.setLayoutManager(
            new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false));
        rvThumbnails.setAdapter(thumbAdapter);

        btnPick.setOnClickListener(v -> pickLauncher.launch("image/*"));
        btnAnalyze.setOnClickListener(v -> startAnalysis());

        executor     = Executors.newSingleThreadExecutor();
        faceDetector = new FaceDetectorHelper();

        try {
            faceEmbedder = new FaceEmbedder(this);
        } catch (RuntimeException e) {
            showError("facenet.tflite not found in assets");
        }

        showPrivacyDialogIfNeeded();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor != null) executor.shutdownNow();
        if (faceDetector != null) faceDetector.close();
        if (faceEmbedder != null) faceEmbedder.close();
    }

    // ── Selection UI ──────────────────────────────────────────────────────────

    private void updateSelectionUI() {
        int n = selectedUris.size();
        tvCount.setText(n + " photo" + (n == 1 ? "" : "s") + " selected");
        ((ThumbnailAdapter) rvThumbnails.getAdapter()).setUris(selectedUris);
        updateAnalyzeButton();
    }

    private void updateAnalyzeButton() {
        int n = selectedUris.size();
        if (n == 0) {
            btnAnalyze.setText("ANALYZE 0 PHOTOS");
            btnAnalyze.setEnabled(false);
        } else {
            btnAnalyze.setText("ANALYZE " + n + " PHOTO" + (n == 1 ? "" : "S"));
            btnAnalyze.setEnabled(faceEmbedder != null);
        }
    }

    private void setUiProcessing(boolean processing) {
        btnPick.setEnabled(!processing);
        btnAnalyze.setEnabled(!processing);
        progressBar.setVisibility(processing ? View.VISIBLE : View.GONE);
        tvProgress.setVisibility(processing ? View.VISIBLE : View.GONE);
        tvError.setVisibility(View.GONE);
    }

    private void showError(String msg) {
        tvError.setText(msg);
        tvError.setVisibility(View.VISIBLE);
    }

    // ── Analysis pipeline ─────────────────────────────────────────────────────

    private void startAnalysis() {
        if (selectedUris.isEmpty() || faceEmbedder == null) return;
        setUiProcessing(true);
        progressBar.setMax(selectedUris.size());
        progressBar.setIndeterminate(false);

        executor.execute(() -> {
            try {
                List<FaceClusterer.FaceEmbedding> allFaces = new ArrayList<>();

                for (int i = 0; i < selectedUris.size(); i++) {
                    final int idx = i;
                    runOnUiThread(() -> {
                        tvProgress.setText("Detecting faces in photo " + (idx + 1)
                            + " / " + selectedUris.size() + "...");
                        progressBar.setProgress(idx + 1);
                    });

                    Uri uri = selectedUris.get(i);
                    Bitmap bmp = decodeBitmap(uri, 1024);
                    if (bmp == null) continue;

                    List<Rect> boxes = faceDetector.detect(bmp);

                    for (int f = 0; f < boxes.size(); f++) {
                        Bitmap crop = cropFace(bmp, boxes.get(f), 0.20f);
                        float[] embedding = faceEmbedder.embed(crop);
                        // Downscale crop for display (save memory)
                        Bitmap thumb = Bitmap.createScaledBitmap(crop, 112, 112, true);
                        if (thumb != crop) crop.recycle();
                        allFaces.add(new FaceClusterer.FaceEmbedding(
                            uri.toString(), thumb, embedding, f));
                    }
                    bmp.recycle();
                }

                List<FaceClusterer.FaceCluster> clusters = FaceClusterer.cluster(allFaces);

                runOnUiThread(() -> showResults(clusters));

            } catch (Exception e) {
                runOnUiThread(() -> {
                    setUiProcessing(false);
                    showError("Error: " + e.getMessage());
                });
            }
        });
    }

    private void showResults(List<FaceClusterer.FaceCluster> clusters) {
        setUiProcessing(false);
        selectionSection.setVisibility(View.GONE);
        resultsSection.setVisibility(View.VISIBLE);

        long personCount = clusters.stream().filter(c -> c.personIndex > 0).count();
        long unknownCount = clusters.stream().filter(c -> c.personIndex == 0).count();
        tvResultsHeader.setText(String.format(Locale.US,
            "%d person group%s found  ·  %d unmatched face%s",
            personCount, personCount == 1 ? "" : "s",
            unknownCount, unknownCount == 1 ? "" : "s"));

        rvGroups.setLayoutManager(new LinearLayoutManager(this));
        rvGroups.setItemAnimator(null);
        rvGroups.setAdapter(new PersonGroupAdapter(clusters));
    }

    // ── Adapters ──────────────────────────────────────────────────────────────

    private class PersonGroupAdapter
        extends RecyclerView.Adapter<PersonGroupAdapter.VH> {

        private final List<FaceClusterer.FaceCluster> clusters;
        private final Set<Integer> expanded = new HashSet<>();

        PersonGroupAdapter(List<FaceClusterer.FaceCluster> clusters) {
            this.clusters = clusters;
        }

        @Override
        public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            android.view.View v = getLayoutInflater()
                .inflate(R.layout.item_face_cluster, parent, false);
            return new VH(v);
        }

        @Override
        public void onBindViewHolder(VH holder, int position) {
            FaceClusterer.FaceCluster cluster = clusters.get(position);

            // Build face crop row (up to 5)
            holder.faceCropRow.removeAllViews();
            int facesToShow = Math.min(cluster.faces.size(), 5);
            int sizePx = (int) (56 * getResources().getDisplayMetrics().density);
            int marginPx = (int) (4 * getResources().getDisplayMetrics().density);

            for (int i = 0; i < facesToShow; i++) {
                ImageView iv = new ImageView(FaceGroupsActivity.this);
                LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(sizePx, sizePx);
                lp.setMargins(0, 0, marginPx, 0);
                iv.setLayoutParams(lp);
                iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
                Glide.with(FaceGroupsActivity.this)
                    .load(cluster.faces.get(i).crop)
                    .circleCrop()
                    .into(iv);
                holder.faceCropRow.addView(iv);
            }

            // "+N more" label
            int extra = cluster.faces.size() - 5;
            if (extra > 0) {
                holder.tvMoreFaces.setText("+" + extra + " more");
                holder.tvMoreFaces.setVisibility(View.VISIBLE);
            } else {
                holder.tvMoreFaces.setVisibility(View.GONE);
            }

            // Person label
            if (cluster.personIndex > 0) {
                holder.tvPersonLabel.setText(String.format(Locale.US,
                    "PERSON %d  ·  %d PHOTO%s",
                    cluster.personIndex,
                    cluster.faces.size(),
                    cluster.faces.size() == 1 ? "" : "S"));
            } else {
                holder.tvPersonLabel.setText(String.format(Locale.US,
                    "UNMATCHED  ·  %d FACE%s",
                    cluster.faces.size(),
                    cluster.faces.size() == 1 ? "" : "S"));
            }

            // Expand/collapse
            boolean isExpanded = expanded.contains(position);
            if (isExpanded) {
                holder.rvPersonPhotos.setVisibility(View.VISIBLE);
                holder.rvPersonPhotos.setLayoutManager(
                    new LinearLayoutManager(FaceGroupsActivity.this,
                        LinearLayoutManager.HORIZONTAL, false));
                holder.rvPersonPhotos.setAdapter(new PersonPhotoAdapter(cluster.faces));
            } else {
                holder.rvPersonPhotos.setVisibility(View.GONE);
            }
            holder.rvPersonPhotos.setNestedScrollingEnabled(false);

            holder.itemView.setOnClickListener(v -> {
                int pos = holder.getAdapterPosition();
                if (pos == RecyclerView.NO_POSITION) return;
                if (expanded.contains(pos)) expanded.remove(pos);
                else                        expanded.add(pos);
                notifyItemChanged(pos);
            });
        }

        @Override
        public int getItemCount() { return clusters.size(); }

        @Override
        public void onViewRecycled(VH holder) {
            super.onViewRecycled(holder);
            holder.faceCropRow.removeAllViews();
        }

        class VH extends RecyclerView.ViewHolder {
            final LinearLayout faceCropRow;
            final TextView     tvMoreFaces;
            final TextView     tvPersonLabel;
            final RecyclerView rvPersonPhotos;
            VH(android.view.View v) {
                super(v);
                faceCropRow    = v.findViewById(R.id.faceCropRow);
                tvMoreFaces    = v.findViewById(R.id.tvMoreFaces);
                tvPersonLabel  = v.findViewById(R.id.tvPersonLabel);
                rvPersonPhotos = v.findViewById(R.id.rvPersonPhotos);
            }
        }
    }

    private class PersonPhotoAdapter
        extends RecyclerView.Adapter<PersonPhotoAdapter.VH> {

        private final List<FaceClusterer.FaceEmbedding> faces;

        PersonPhotoAdapter(List<FaceClusterer.FaceEmbedding> faces) {
            this.faces = faces;
        }

        @Override
        public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            ImageView iv = (ImageView) getLayoutInflater()
                .inflate(R.layout.item_person_photo, parent, false);
            return new VH(iv);
        }

        @Override
        public void onBindViewHolder(VH holder, int position) {
            Glide.with(FaceGroupsActivity.this)
                .load(Uri.parse(faces.get(position).photoUri))
                .centerCrop()
                .into(holder.iv);
        }

        @Override public int getItemCount() { return faces.size(); }

        @Override
        public void onViewRecycled(VH holder) {
            super.onViewRecycled(holder);
            Glide.with(FaceGroupsActivity.this).clear(holder.iv);
        }

        class VH extends RecyclerView.ViewHolder {
            final ImageView iv;
            VH(ImageView iv) { super(iv); this.iv = iv; }
        }
    }

    private class ThumbnailAdapter extends RecyclerView.Adapter<ThumbnailAdapter.VH> {

        private List<Uri> uris = new ArrayList<>();

        void setUris(List<Uri> uris) {
            this.uris = uris;
            notifyDataSetChanged();
        }

        @Override
        public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            ImageView iv = new ImageView(parent.getContext());
            int sizePx   = (int) (80 * getResources().getDisplayMetrics().density);
            int marginPx = (int) (2  * getResources().getDisplayMetrics().density);
            RecyclerView.LayoutParams lp = new RecyclerView.LayoutParams(sizePx, sizePx);
            lp.setMargins(marginPx, marginPx, marginPx, marginPx);
            iv.setLayoutParams(lp);
            iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
            return new VH(iv);
        }

        @Override
        public void onBindViewHolder(VH holder, int position) {
            Glide.with(FaceGroupsActivity.this).load(uris.get(position)).centerCrop().into(holder.iv);
        }

        @Override public int getItemCount() { return uris.size(); }

        class VH extends RecyclerView.ViewHolder {
            final ImageView iv;
            VH(ImageView iv) { super(iv); this.iv = iv; }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private Bitmap decodeBitmap(Uri uri, int maxSide) {
        try {
            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            try (InputStream is = getContentResolver().openInputStream(uri)) {
                BitmapFactory.decodeStream(is, null, opts);
            }
            opts.inSampleSize = MainActivity.computeSampleSize(opts.outWidth, opts.outHeight, maxSide);
            opts.inJustDecodeBounds = false;
            try (InputStream is = getContentResolver().openInputStream(uri)) {
                return BitmapFactory.decodeStream(is, null, opts);
            }
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Crops a face from the source bitmap with proportional padding around the bounding box.
     */
    private Bitmap cropFace(Bitmap src, Rect box, float padding) {
        int padX = (int) (box.width()  * padding);
        int padY = (int) (box.height() * padding);
        int left   = Math.max(0, box.left   - padX);
        int top    = Math.max(0, box.top    - padY);
        int right  = Math.min(src.getWidth(),  box.right  + padX);
        int bottom = Math.min(src.getHeight(), box.bottom + padY);
        return Bitmap.createBitmap(src, left, top, right - left, bottom - top);
    }

    private void showPrivacyDialogIfNeeded() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        if (prefs.getBoolean(KEY_PRIVACY_SHOWN, false)) return;
        new AlertDialog.Builder(this)
            .setTitle("On-Device Only")
            .setMessage("Face analysis runs entirely on your device. No data is uploaded.")
            .setPositiveButton("OK", (d, w) ->
                prefs.edit().putBoolean(KEY_PRIVACY_SHOWN, true).apply())
            .setCancelable(false)
            .show();
    }
}
