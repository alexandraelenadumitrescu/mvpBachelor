package com.photomatch;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.photomatch.api.ApiClient;
import com.photomatch.api.ApiService;
import com.photomatch.api.ApplyLutRequest;
import com.photomatch.api.ApplyLutResponse;
import com.photomatch.api.BatchResponse;
import com.photomatch.api.SearchAndCorrectRequest;
import com.photomatch.api.SearchAndCorrectResponse;
import com.photomatch.api.StyleSearchRequest;
import com.photomatch.api.StyleSearchResponse;
import com.photomatch.ml.CLIPEncoder;
import com.photomatch.ml.DefectDetector;
import com.photomatch.ml.HybridVectorBuilder;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Response;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class PipelineActivity extends AppCompatActivity {

    private static final int    MAX_PHOTOS = 50;
    private static final String PREFS      = "photomatch_prefs";
    private static final String KEY_SID    = "style_session_id";

    private final List<Uri>          selectedUris = new ArrayList<>();
    private final List<PipelineStep> steps        = new ArrayList<>();

    private PipelineAdapter adapter;
    private LinearLayout    chipContainer;
    private TextView        tvPhotoCount;
    private Button          btnRun;
    private RecyclerView    rvSteps;
    private LinearLayout    emptyState;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    // Lazy-init only if the corresponding step is used
    private CLIPEncoder        clipEncoder;
    private DefectDetector     defectDetector;
    private FaceDetectorHelper faceDetector;
    private FaceEmbedder       faceEmbedder;

    // Per-run vector caches: cleared at start of each runPipeline()
    private final Map<Uri, float[]> clipCache   = new HashMap<>();
    private final Map<Uri, float[]> defectCache = new HashMap<>();

    private final ActivityResultLauncher<String> pickLauncher =
        registerForActivityResult(new ActivityResultContracts.GetMultipleContents(), uris -> {
            if (uris != null && !uris.isEmpty()) {
                selectedUris.clear();
                selectedUris.addAll(uris.subList(0, Math.min(uris.size(), MAX_PHOTOS)));
                int n = selectedUris.size();
                tvPhotoCount.setText(n + " photo" + (n == 1 ? "" : "s") + " selected");
                updateRunButton();
            }
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pipeline);

        tvPhotoCount  = findViewById(R.id.tvPhotoCount);
        chipContainer = findViewById(R.id.chipContainer);
        rvSteps       = findViewById(R.id.rvSteps);
        emptyState    = findViewById(R.id.emptyState);
        btnRun        = findViewById(R.id.btnRun);

        findViewById(R.id.btnPickPhotos).setOnClickListener(v -> pickLauncher.launch("image/*"));

        // Build step-type chip buttons
        float density = getResources().getDisplayMetrics().density;
        int padPx = (int) (8 * density);
        for (PipelineStep.Type type : PipelineStep.Type.values()) {
            Button chip = new Button(this);
            chip.setText(chipLabel(type));
            chip.setBackgroundColor(Color.TRANSPARENT);
            chip.setTextColor(Color.parseColor("#C9A84C"));
            chip.setTextSize(11);
            chip.setTypeface(android.graphics.Typeface.MONOSPACE);
            chip.setLetterSpacing(0.05f);
            chip.setPadding(padPx, 0, padPx, 0);
            chip.setMinWidth(0);
            chip.setMinimumWidth(0);
            chip.setOnClickListener(v -> addStep(type));
            chipContainer.addView(chip);
        }

        // RecyclerView + adapter
        adapter = new PipelineAdapter(steps);
        adapter.setOnDeleteListener(pos -> {
            steps.remove(pos);
            adapter.notifyItemRemoved(pos);
            adapter.notifyItemRangeChanged(pos, steps.size());
            updateEmptyState();
            updateRunButton();
        });
        rvSteps.setLayoutManager(new LinearLayoutManager(this));
        rvSteps.setAdapter(adapter);
        rvSteps.setItemAnimator(null);

        // Drag-to-reorder
        ItemTouchHelper.SimpleCallback dragCb = new ItemTouchHelper.SimpleCallback(
                ItemTouchHelper.UP | ItemTouchHelper.DOWN, 0) {
            @Override
            public boolean onMove(RecyclerView rv,
                                  RecyclerView.ViewHolder from,
                                  RecyclerView.ViewHolder to) {
                adapter.moveItem(from.getAdapterPosition(), to.getAdapterPosition());
                return true;
            }
            @Override public void onSwiped(RecyclerView.ViewHolder vh, int dir) {}
            @Override
            public void onSelectedChanged(RecyclerView.ViewHolder vh, int state) {
                super.onSelectedChanged(vh, state);
                if (state == ItemTouchHelper.ACTION_STATE_DRAG && vh != null)
                    vh.itemView.setAlpha(0.7f);
            }
            @Override
            public void clearView(RecyclerView rv, RecyclerView.ViewHolder vh) {
                super.clearView(rv, vh);
                vh.itemView.setAlpha(1.0f);
                adapter.notifyDataSetChanged(); // refresh step numbers after drop
            }
        };
        new ItemTouchHelper(dragCb).attachToRecyclerView(rvSteps);

        btnRun.setOnClickListener(v -> runPipeline());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
        if (clipEncoder    != null) clipEncoder.close();
        if (defectDetector != null) defectDetector.close();
        if (faceDetector   != null) faceDetector.close();
        if (faceEmbedder   != null) faceEmbedder.close();
    }

    // ── Step management ───────────────────────────────────────────────────────

    private void addStep(PipelineStep.Type type) {
        steps.add(new PipelineStep(type));
        adapter.notifyItemInserted(steps.size() - 1);
        // Refresh previous last item so its connector line becomes visible
        if (steps.size() > 1) adapter.notifyItemChanged(steps.size() - 2);
        updateEmptyState();
        updateRunButton();
    }

    private void updateEmptyState() {
        boolean empty = steps.isEmpty();
        emptyState.setVisibility(empty ? View.VISIBLE : View.GONE);
        rvSteps.setVisibility(empty ? View.GONE : View.VISIBLE);
    }

    private void updateRunButton() {
        btnRun.setEnabled(!selectedUris.isEmpty() && !steps.isEmpty());
    }

    public void resetPipeline() {
        steps.clear();
        adapter.notifyDataSetChanged();
        selectedUris.clear();
        tvPhotoCount.setText("0 photos selected");
        PipelineResultsHolder.instance.clear();
        updateEmptyState();
        updateRunButton();
    }

    // ── Pipeline execution ────────────────────────────────────────────────────

    private void runPipeline() {
        if (selectedUris.isEmpty() || steps.isEmpty()) return;

        // Reset all step statuses + result data
        for (PipelineStep s : steps) {
            s.status = PipelineStep.Status.IDLE;
            s.summary = "";
            s.resultDetail = "";
            s.resultUris.clear();
            s.resultBase64.clear();
        }
        adapter.notifyDataSetChanged();

        setChipsEnabled(false);
        btnRun.setEnabled(false);

        executor.execute(() -> {
            // Clear per-run vector caches
            clipCache.clear();
            defectCache.clear();

            List<Uri> workingSet = new ArrayList<>(selectedUris);
            int errorIdx = -1;

            for (int i = 0; i < steps.size(); i++) {
                final int idx = i;
                final PipelineStep step = steps.get(idx);

                runOnUiThread(() -> {
                    step.status = PipelineStep.Status.RUNNING;
                    adapter.notifyItemChanged(idx);
                });

                try {
                    String summary = executeStep(step, workingSet);
                    runOnUiThread(() -> {
                        step.status  = PipelineStep.Status.DONE;
                        step.summary = summary;
                        adapter.notifyItemChanged(idx);
                    });
                } catch (Exception e) {
                    String msg = e.getMessage() != null ? e.getMessage() : "unknown error";
                    runOnUiThread(() -> {
                        step.status  = PipelineStep.Status.ERROR;
                        step.summary = msg;
                        adapter.notifyItemChanged(idx);
                    });
                    errorIdx = idx;
                    break;
                }
            }

            // Build results holder from final working set
            PipelineResultsHolder.instance.clear();
            for (Uri uri : workingSet) {
                PipelineResultsHolder.PipelinePhoto p = new PipelineResultsHolder.PipelinePhoto();
                p.originalUri    = uri;
                p.correctedBase64 = findCorrectedBase64(uri, steps);
                PipelineResultsHolder.instance.photos.add(p);
            }

            final boolean finished = (errorIdx < 0);
            runOnUiThread(() -> {
                setChipsEnabled(true);
                updateRunButton();
                if (finished) showResultsBottomSheet();
            });
        });
    }

    private String executeStep(PipelineStep step, List<Uri> workingSet) throws Exception {
        switch (step.type) {
            case BLUR_FILTER:    return stepBlurFilter(step, workingSet);
            case BURST_DETECT:   return stepBurstDetect(step, workingSet);
            case FACE_GROUP:     return stepFaceGroup(step, workingSet);
            case COLOR_CORRECT:  return stepColorCorrect(step, workingSet);
            case STYLE_TRANSFER: return stepStyleTransfer(step, workingSet);
            case BATCH_EDIT:     return stepBatchEdit(step, workingSet);
            default: throw new IllegalStateException("Unknown step type");
        }
    }

    // ── Vector helpers ────────────────────────────────────────────────────────

    private void ensureClipEncoder() throws Exception {
        if (clipEncoder != null) return;
        clipEncoder = new CLIPEncoder(this);
        CountDownLatch latch = new CountDownLatch(1);
        Exception[] err = {null};
        clipEncoder.loadAsync(Executors.newSingleThreadExecutor(), new CLIPEncoder.LoadCallback() {
            @Override public void onLoaded() { latch.countDown(); }
            @Override public void onError(Exception e) { err[0] = e; latch.countDown(); }
        });
        latch.await();
        if (err[0] != null) throw err[0];
    }

    /** Lazily compute and cache CLIP + defect vectors for a URI. */
    private void ensureVectors(Uri uri, Bitmap bmp) throws Exception {
        if (!clipCache.containsKey(uri)) {
            ensureClipEncoder();
            clipCache.put(uri, clipEncoder.encode(bmp));
        }
        if (!defectCache.containsKey(uri)) {
            if (defectDetector == null) defectDetector = new DefectDetector(this);
            defectCache.put(uri, defectDetector.detect(bmp));
        }
    }

    /** Returns 517-dim hybrid vector from cache, or null if not yet computed. */
    private float[] getHybridVector(Uri uri) {
        float[] clip   = clipCache.get(uri);
        float[] defect = defectCache.get(uri);
        if (clip == null || defect == null) return null;
        return HybridVectorBuilder.build(clip, defect);
    }

    // ── Step implementations ──────────────────────────────────────────────────

    private String stepBlurFilter(PipelineStep step, List<Uri> workingSet) throws IOException {
        int removed = 0;
        List<Uri> keep = new ArrayList<>();
        List<Uri> removed_uris = new ArrayList<>();
        for (Uri uri : workingSet) {
            Bitmap bmp = decodeBitmap(uri, 512);
            if (bmp == null) { keep.add(uri); continue; }
            boolean blurry = BlurDetector.check(bmp, step.strengthParam).isBlurry;
            bmp.recycle();
            if (!blurry) keep.add(uri);
            else { removed_uris.add(uri); removed++; }
        }
        step.resultUris.addAll(removed_uris);
        step.resultDetail = workingSet.size() + " photos checked, " + removed + " removed";
        workingSet.clear();
        workingSet.addAll(keep);
        return "removed " + removed + " blurry photo(s), " + workingSet.size() + " remaining";
    }

    private String stepBurstDetect(PipelineStep step, List<Uri> workingSet) throws Exception {
        ensureClipEncoder();

        List<float[]> vectors = new ArrayList<>();
        for (Uri uri : workingSet) {
            Bitmap bmp = decodeBitmap(uri, 512);
            if (bmp == null) { vectors.add(new float[512]); continue; }
            ensureVectors(uri, bmp);
            bmp.recycle();
            vectors.add(clipCache.get(uri));
        }

        List<List<Integer>> clusters = BurstClusterer.cluster(vectors, step.strengthParam);

        List<Uri> keep = new ArrayList<>();
        for (List<Integer> cluster : clusters) keep.add(workingSet.get(cluster.get(0)));

        int duplicates = workingSet.size() - keep.size();
        step.resultUris.addAll(keep);
        step.resultDetail = (duplicates) + " duplicate(s) hidden, "
            + keep.size() + " kept";

        workingSet.clear();
        workingSet.addAll(keep);
        return "found " + clusters.size() + " cluster(s), kept " + workingSet.size() + " photo(s)";
    }

    private String stepFaceGroup(PipelineStep step, List<Uri> workingSet) throws Exception {
        if (faceDetector == null) faceDetector = new FaceDetectorHelper();
        if (faceEmbedder == null) faceEmbedder = new FaceEmbedder(this);

        List<FaceClusterer.FaceEmbedding> allFaces = new ArrayList<>();
        for (Uri uri : workingSet) {
            Bitmap bmp = decodeBitmap(uri, 1024);
            if (bmp == null) continue;
            List<Rect> boxes = faceDetector.detect(bmp);
            for (int f = 0; f < boxes.size(); f++) {
                Bitmap crop = cropFace(bmp, boxes.get(f), 0.2f);
                float[] emb = faceEmbedder.embed(crop);
                crop.recycle();
                allFaces.add(new FaceClusterer.FaceEmbedding(uri.toString(), null, emb, f));
            }
            bmp.recycle();
        }

        List<FaceClusterer.FaceCluster> clusters = FaceClusterer.cluster(allFaces, step.strengthParam);
        long personGroups = clusters.stream().filter(c -> c.personIndex > 0).count();
        step.resultDetail = "Across " + workingSet.size() + " photos";
        return "found " + personGroups + " person group(s) across " + allFaces.size() + " face(s)";
    }

    private String stepColorCorrect(PipelineStep step, List<Uri> workingSet) {
        ApiService api = ApiClient.getInstance().getService();
        int ok = 0, fail = 0;

        for (Uri uri : workingSet) {
            try {
                // 1. Ensure vectors are computed
                Bitmap bmp = decodeBitmap(uri, 512);
                if (bmp != null) { ensureVectors(uri, bmp); bmp.recycle(); }

                float[] hybrid = getHybridVector(uri);
                if (hybrid == null) { fail++; continue; }

                // 2. FAISS retrieval — no image upload, no server CLIP inference
                SearchAndCorrectRequest req = new SearchAndCorrectRequest(hybrid);
                Response<SearchAndCorrectResponse> searchResp =
                    api.searchAndCorrect(req, step.strengthParam, false).execute();
                if (!searchResp.isSuccessful() || searchResp.body() == null) { fail++; continue; }
                String basename = searchResp.body().retrieved;

                // 3. Apply correction — upload image once
                ApplyLutRequest lutReq = new ApplyLutRequest();
                lutReq.imageB64          = Base64.encodeToString(readBytes(uri), Base64.NO_WRAP);
                lutReq.retrievedBasename = basename;
                Response<ApplyLutResponse> lutResp = api.applyLut(lutReq).execute();
                if (!lutResp.isSuccessful() || lutResp.body() == null) { fail++; continue; }

                step.resultUris.add(uri);
                step.resultBase64.add(blendBase64(uri, lutResp.body().finalB64, step.strengthParam));
                ok++;
            } catch (Exception e) {
                Log.e("Pipeline", "COLOR_CORRECT error: " + e.getMessage());
                fail++;
            }
        }
        step.resultDetail = ok + " photo(s) color-corrected";
        return ok + " corrected" + (fail > 0 ? ", " + fail + " failed" : "");
    }

    private String stepStyleTransfer(PipelineStep step, List<Uri> workingSet) {
        String sessionId = getSharedPreferences(PREFS, MODE_PRIVATE).getString(KEY_SID, null);
        if (sessionId == null) return "no style set up \u2014 skipped";

        ApiService api = ApiClient.getInstance().getService();
        int ok = 0, fail = 0;

        for (Uri uri : workingSet) {
            try {
                // 1. Ensure vectors are computed
                Bitmap bmp = decodeBitmap(uri, 512);
                if (bmp != null) { ensureVectors(uri, bmp); bmp.recycle(); }

                float[] hybrid = getHybridVector(uri);
                if (hybrid == null) { fail++; continue; }

                // 2. Style-constrained FAISS retrieval
                StyleSearchRequest req = new StyleSearchRequest(hybrid, sessionId);
                Response<StyleSearchResponse> searchResp = api.styleSearch(req).execute();
                if (!searchResp.isSuccessful() || searchResp.body() == null) { fail++; continue; }
                String basename = searchResp.body().retrieved;

                // 3. Apply correction
                ApplyLutRequest lutReq = new ApplyLutRequest();
                lutReq.imageB64          = Base64.encodeToString(readBytes(uri), Base64.NO_WRAP);
                lutReq.retrievedBasename = basename;
                Response<ApplyLutResponse> lutResp = api.applyLut(lutReq).execute();
                if (!lutResp.isSuccessful() || lutResp.body() == null) { fail++; continue; }

                step.resultUris.add(uri);
                step.resultBase64.add(blendBase64(uri, lutResp.body().finalB64, step.strengthParam));
                ok++;
            } catch (Exception e) {
                Log.e("Pipeline", "STYLE_TRANSFER error: " + e.getMessage());
                fail++;
            }
        }
        step.resultDetail = ok + " photo(s) styled";
        return ok + " styled" + (fail > 0 ? ", " + fail + " failed" : "");
    }

    private String stepBatchEdit(PipelineStep step, List<Uri> workingSet) {
        if (workingSet.isEmpty()) return "no photos to process";
        ApiService api = ApiClient.getInstance().getService();
        try {
            List<MultipartBody.Part> parts = new ArrayList<>();
            for (Uri uri : workingSet) {
                byte[] bytes = readBytes(uri);
                RequestBody rb = RequestBody.create(MediaType.parse("image/jpeg"), bytes);
                parts.add(MultipartBody.Part.createFormData("files", "image.jpg", rb));
            }
            Response<BatchResponse> resp = api.batchProcess(parts).execute();
            if (resp.isSuccessful()) {
                BatchResponse body = resp.body();
                int count = (body != null) ? body.processed : workingSet.size();
                step.resultUris.addAll(workingSet);
                step.resultDetail = count + " photo(s) batch-processed";
                return "batch-processed " + count + " photo(s)";
            } else {
                return "server error: " + resp.code();
            }
        } catch (IOException e) {
            return "network error: " + (e.getMessage() != null ? e.getMessage() : "unreachable");
        }
    }

    // ── UI helpers ─────────────────────────────────────────────────────────────

    private void setChipsEnabled(boolean enabled) {
        for (int i = 0; i < chipContainer.getChildCount(); i++)
            chipContainer.getChildAt(i).setEnabled(enabled);
    }

    private void showResultsBottomSheet() {
        new PipelineResultsBottomSheet()
            .show(getSupportFragmentManager(), "results");
    }

    // ── Image helpers ──────────────────────────────────────────────────────────

    private Bitmap decodeBitmap(Uri uri, int maxSide) {
        try {
            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            try (InputStream is = getContentResolver().openInputStream(uri)) {
                BitmapFactory.decodeStream(is, null, opts);
            }
            opts.inSampleSize =
                MainActivity.computeSampleSize(opts.outWidth, opts.outHeight, maxSide);
            opts.inJustDecodeBounds = false;
            try (InputStream is = getContentResolver().openInputStream(uri)) {
                return BitmapFactory.decodeStream(is, null, opts);
            }
        } catch (IOException e) {
            return null;
        }
    }

    private byte[] readBytes(Uri uri) throws IOException {
        try (InputStream is = getContentResolver().openInputStream(uri);
             ByteArrayOutputStream buf = new ByteArrayOutputStream()) {
            byte[] tmp = new byte[8192];
            int n;
            while ((n = is.read(tmp)) != -1) buf.write(tmp, 0, n);
            return buf.toByteArray();
        }
    }

    private Bitmap cropFace(Bitmap src, Rect box, float padding) {
        int padX = (int) (box.width()  * padding);
        int padY = (int) (box.height() * padding);
        int l = Math.max(0, box.left   - padX);
        int t = Math.max(0, box.top    - padY);
        int r = Math.min(src.getWidth(),  box.right  + padX);
        int b = Math.min(src.getHeight(), box.bottom + padY);
        return Bitmap.createBitmap(src, l, t, r - l, b - t);
    }

    /** Blend orig*(1-s) + corrected*s and return as base64 JPEG. */
    private String blendBase64(Uri origUri, String correctedB64, float s) {
        Bitmap orig      = decodeBitmap(origUri, 512);
        Bitmap corrected = ApiClient.base64ToBitmap(correctedB64);
        if (orig == null || corrected == null) return correctedB64;

        Bitmap scaled = Bitmap.createScaledBitmap(corrected, orig.getWidth(), orig.getHeight(), true);
        Bitmap result = orig.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(result);
        Paint  paint  = new Paint();
        paint.setAlpha((int) (s * 255));
        canvas.drawBitmap(scaled, 0, 0, paint);
        scaled.recycle();
        orig.recycle();
        corrected.recycle();

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        result.compress(Bitmap.CompressFormat.JPEG, 85, out);
        result.recycle();
        return Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP);
    }

    /** Find the latest corrected base64 for a URI across all DONE steps. */
    private String findCorrectedBase64(Uri uri, List<PipelineStep> steps) {
        for (int i = steps.size() - 1; i >= 0; i--) {
            PipelineStep s = steps.get(i);
            if (s.status != PipelineStep.Status.DONE) continue;
            int idx = s.resultUris.indexOf(uri);
            if (idx >= 0 && idx < s.resultBase64.size() && s.resultBase64.get(idx) != null) {
                return s.resultBase64.get(idx);
            }
        }
        return null;
    }

    private static String chipLabel(PipelineStep.Type type) {
        switch (type) {
            case BLUR_FILTER:    return "BLUR FILTER";
            case COLOR_CORRECT:  return "COLOR CORRECT";
            case BURST_DETECT:   return "BURST DETECT";
            case FACE_GROUP:     return "FACE GROUP";
            case STYLE_TRANSFER: return "STYLE TRANSFER";
            default:             return "BATCH EDIT";
        }
    }
}
