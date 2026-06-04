package com.photomatch;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.util.Base64;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.core.content.FileProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.photomatch.api.BlurRegion;
import com.photomatch.api.ScrapeRequest;
import com.photomatch.api.ScrapeResponse;
import com.photomatch.api.ScrapedImageResult;
import com.photomatch.base.BaseServerActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class WebScraperActivity extends BaseServerActivity {

    private EditText     etUrl;
    private Button       btnScrape;
    private RecyclerView rvResults;
    private TextView     tvStatus;
    private TextView     tvError;
    private ProgressBar  progressBar;
    private Button[]     filterButtons;

    private final List<ScrapedImageResult> allResults       = new ArrayList<>();
    private final List<ScrapedImageResult> displayedResults = new ArrayList<>();
    private ScrapeResultAdapter adapter;
    private String activeFilter = "TOATE";

    private static final String[] FILTERS = {"TOATE", "HIGH", "MEDIUM", "LOW"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_web_scraper);

        etUrl       = findViewById(R.id.etUrl);
        btnScrape   = primaryButton = findViewById(R.id.btnScrape);
        rvResults   = findViewById(R.id.rvResults);
        tvStatus    = findViewById(R.id.tvStatus);
        tvError     = findViewById(R.id.tvError);
        progressBar = findViewById(R.id.progressBar);

        filterButtons = new Button[]{
            findViewById(R.id.btnFilterAll),
            findViewById(R.id.btnFilterHigh),
            findViewById(R.id.btnFilterMedium),
            findViewById(R.id.btnFilterLow)
        };
        for (int i = 0; i < FILTERS.length; i++) {
            final String filter = FILTERS[i];
            filterButtons[i].setTag(filter);
            filterButtons[i].setOnClickListener(v -> setFilter(filter));
        }
        updateFilterButtonStyles();

        adapter = new ScrapeResultAdapter(displayedResults, new OnItemAction() {
            @Override public void onOpenBlurEditor(ScrapedImageResult item) { openInBlurEditor(item); }
            @Override public void onApplyBlurInline(ScrapedImageResult item) { applyBlurInline(item); }
        });
        rvResults.setLayoutManager(new LinearLayoutManager(this));
        rvResults.setAdapter(adapter);

        btnScrape.setOnClickListener(v -> startScrape());
    }

    // ── Scrape ────────────────────────────────────────────────────────────────

    private void startScrape() {
        String url = etUrl.getText().toString().trim();
        if (url.isEmpty()) { showError("Introdu un URL"); return; }
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            showError("URL-ul trebuie sa inceapa cu http:// sau https://");
            return;
        }

        setProcessing(true);
        tvStatus.setVisibility(View.GONE);
        tvError.setVisibility(View.GONE);
        allResults.clear();
        displayedResults.clear();
        adapter.clearBlurState();
        adapter.notifyDataSetChanged();

        api().scrapeImages(new ScrapeRequest(url, "yolo")).enqueue(new Callback<ScrapeResponse>() {
            @Override
            public void onResponse(Call<ScrapeResponse> call, Response<ScrapeResponse> response) {
                setProcessing(false);
                if (!response.isSuccessful() || response.body() == null) {
                    showError("Eroare server: HTTP " + response.code());
                    return;
                }
                ScrapeResponse body = response.body();
                if (body.results != null) {
                    for (ScrapedImageResult r : body.results) {
                        if (r.riskScore > 0) allResults.add(r);
                    }
                }
                applyFilter();
                tvStatus.setText(String.format(Locale.US,
                    "%d imagini riscante din %d procesate (%d sarite, max 30)",
                    body.imagesWithRisk, body.imagesProcessed, body.skippedCount));
                tvStatus.setVisibility(View.VISIBLE);
            }

            @Override
            public void onFailure(Call<ScrapeResponse> call, Throwable t) {
                setProcessing(false);
                showError("Eroare retea: " + t.getMessage());
            }
        });
    }

    // ── Filter ────────────────────────────────────────────────────────────────

    private void setFilter(String filter) {
        activeFilter = filter;
        updateFilterButtonStyles();
        applyFilter();
    }

    private void applyFilter() {
        displayedResults.clear();
        for (ScrapedImageResult r : allResults) {
            if ("TOATE".equals(activeFilter) || activeFilter.equals(r.riskLabel))
                displayedResults.add(r);
        }
        adapter.notifyDataSetChanged();
    }

    private void updateFilterButtonStyles() {
        for (Button b : filterButtons) {
            boolean sel = b.getTag().equals(activeFilter);
            b.setBackgroundColor(sel ? Color.parseColor("#C9A84C") : Color.TRANSPARENT);
            b.setTextColor(sel ? Color.BLACK : Color.parseColor("#C9A84C"));
        }
    }

    // ── Blur ──────────────────────────────────────────────────────────────────

    /** Stable cache filename derived from the image's source URL. */
    private File getCacheFile(ScrapedImageResult item) {
        int hash = item.sourceUrl.hashCode() & 0x7FFFFFFF;
        return new File(getCacheDir(), "scrape_thumb_" + hash + ".jpg");
    }

    /** Writes thumbnail bytes to the cache file only if it doesn't already exist. */
    private void ensureCacheFile(ScrapedImageResult item) throws IOException {
        File f = getCacheFile(item);
        if (!f.exists()) {
            byte[] bytes = Base64.decode(item.thumbnailB64, Base64.DEFAULT);
            try (FileOutputStream fos = new FileOutputStream(f)) { fos.write(bytes); }
        }
    }

    /**
     * Opens the cache file in BlurActivity. Always reads from the cache file so that
     * if applyBlurInline() already ran and wrote blurred bytes there, the editor shows
     * the blurred version rather than the original thumbnail.
     */
    private void openInBlurEditor(ScrapedImageResult item) {
        try {
            ensureCacheFile(item);
            Uri uri = FileProvider.getUriForFile(this, "com.photomatch.fileprovider", getCacheFile(item));
            ArrayList<Uri> uris = new ArrayList<>();
            uris.add(uri);
            Intent intent = new Intent(this, BlurActivity.class);
            intent.putParcelableArrayListExtra(BlurActivity.EXTRA_URIS, uris);
            startActivity(intent);
        } catch (Exception e) {
            showError("Nu se poate deschide: " + e.getMessage());
        }
    }

    /**
     * Calls /blur-sensitive with the cached thumbnail, updates the thumbnail in-place,
     * and overwrites the cache file with the blurred bytes so openInBlurEditor stays consistent.
     */
    private void applyBlurInline(ScrapedImageResult item) {
        adapter.setBlurPending(item.sourceUrl, true);
        int idx = findDisplayedIndex(item);
        if (idx >= 0) adapter.notifyItemChanged(idx);

        executor.execute(() -> {
            try {
                ensureCacheFile(item);
                byte[] imageBytes = Files.readAllBytes(getCacheFile(item).toPath());

                RequestBody rb = RequestBody.create(MediaType.parse("image/jpeg"), imageBytes);
                MultipartBody.Part part = MultipartBody.Part.createFormData("file", "img.jpg", rb);

                Response<ResponseBody> resp = api().blurSensitive(part, "local").execute();
                if (resp.isSuccessful() && resp.body() != null) {
                    byte[] blurred = resp.body().bytes();
                    try (FileOutputStream fos = new FileOutputStream(getCacheFile(item))) {
                        fos.write(blurred);
                    }
                    Bitmap bmp = BitmapFactory.decodeByteArray(blurred, 0, blurred.length);
                    runOnUiThread(() -> {
                        adapter.setBlurResult(item.sourceUrl, bmp);
                        int i = findDisplayedIndex(item);
                        if (i >= 0) adapter.notifyItemChanged(i);
                    });
                } else {
                    runOnUiThread(() -> {
                        adapter.setBlurPending(item.sourceUrl, false);
                        int i = findDisplayedIndex(item);
                        if (i >= 0) adapter.notifyItemChanged(i);
                        showError("Blur esuat: HTTP " + resp.code());
                    });
                }
            } catch (Exception e) {
                runOnUiThread(() -> {
                    adapter.setBlurPending(item.sourceUrl, false);
                    int i = findDisplayedIndex(item);
                    if (i >= 0) adapter.notifyItemChanged(i);
                    showError("Blur esuat: " + e.getMessage());
                });
            }
        });
    }

    private int findDisplayedIndex(ScrapedImageResult target) {
        for (int i = 0; i < displayedResults.size(); i++) {
            if (displayedResults.get(i).sourceUrl.equals(target.sourceUrl)) return i;
        }
        return -1;
    }

    // ── Adapter ───────────────────────────────────────────────────────────────

    interface OnItemAction {
        void onOpenBlurEditor(ScrapedImageResult item);
        void onApplyBlurInline(ScrapedImageResult item);
    }

    static class ScrapeResultAdapter extends RecyclerView.Adapter<ScrapeResultAdapter.VH> {

        private final List<ScrapedImageResult> items;
        private final OnItemAction             listener;
        private final Map<String, Bitmap>      blurredBitmaps = new HashMap<>();
        private final Set<String>              blurPending    = new HashSet<>();

        ScrapeResultAdapter(List<ScrapedImageResult> items, OnItemAction listener) {
            this.items    = items;
            this.listener = listener;
        }

        void setBlurPending(String key, boolean pending) {
            if (pending) blurPending.add(key); else blurPending.remove(key);
        }

        void setBlurResult(String key, Bitmap bmp) {
            blurPending.remove(key);
            blurredBitmaps.put(key, bmp);
        }

        void clearBlurState() {
            blurredBitmaps.clear();
            blurPending.clear();
        }

        @Override
        public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            View v = android.view.LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_scrape_result, parent, false);
            return new VH(v);
        }

        @Override
        public void onBindViewHolder(VH holder, int position) {
            ScrapedImageResult item    = items.get(position);
            String             key     = item.sourceUrl;
            boolean            pending = blurPending.contains(key);
            Bitmap             blurred = blurredBitmaps.get(key);

            // Thumbnail: show blurred result if available, else original with region overlay
            if (blurred != null) {
                holder.thumbnail.setImageBitmap(blurred);
            } else {
                try {
                    byte[] bytes = Base64.decode(item.thumbnailB64, Base64.DEFAULT);
                    Bitmap bmp   = BitmapFactory.decodeByteArray(bytes, 0, bytes.length)
                                               .copy(Bitmap.Config.ARGB_8888, true);
                    if (item.regions != null && !item.regions.isEmpty()
                            && item.originalWidth > 0 && item.thumbnailWidth > 0) {
                        bmp = drawRegionOverlay(bmp, item);
                    }
                    holder.thumbnail.setImageBitmap(bmp);
                } catch (Exception ignored) {
                    holder.thumbnail.setImageResource(android.R.drawable.ic_menu_gallery);
                }
            }

            // Risk label
            holder.riskLabel.setText(item.riskLabel);
            switch (item.riskLabel) {
                case "HIGH":   holder.riskLabel.setBackgroundColor(Color.parseColor("#FF5555")); break;
                case "MEDIUM": holder.riskLabel.setBackgroundColor(Color.parseColor("#FFB300")); break;
                case "LOW":    holder.riskLabel.setBackgroundColor(Color.parseColor("#81C784")); break;
                default:       holder.riskLabel.setBackgroundColor(Color.TRANSPARENT);
            }

            int nRegions = item.regions != null ? item.regions.size() : 0;
            holder.regionCount.setText(nRegions + " regiune" + (nRegions == 1 ? "" : "i") + " detectata");
            holder.sourceUrl.setText(item.sourceUrl);

            // Inline blur state
            holder.blurProgress.setVisibility(pending ? View.VISIBLE : View.GONE);
            holder.btnBlurNow.setVisibility(blurred != null ? View.GONE : View.VISIBLE);
            holder.btnBlurNow.setEnabled(!pending);
            holder.btnBlurNow.setOnClickListener(v -> listener.onApplyBlurInline(item));
            holder.btnOpenBlur.setOnClickListener(v -> listener.onOpenBlurEditor(item));
        }

        /** Draws amber bounding boxes scaled from original-image coordinates to thumbnail size. */
        private Bitmap drawRegionOverlay(Bitmap src, ScrapedImageResult item) {
            float sx = (float) item.thumbnailWidth  / item.originalWidth;
            float sy = (float) item.thumbnailHeight / item.originalHeight;
            Paint paint = new Paint();
            paint.setStyle(Paint.Style.STROKE);
            paint.setColor(Color.parseColor("#C9A84C"));
            paint.setStrokeWidth(3f);
            Canvas canvas = new Canvas(src);
            for (BlurRegion r : item.regions) {
                canvas.drawRect(r.x * sx, r.y * sy, (r.x + r.w) * sx, (r.y + r.h) * sy, paint);
            }
            return src;
        }

        @Override
        public int getItemCount() { return items.size(); }

        @Override
        public void onViewRecycled(VH holder) {
            super.onViewRecycled(holder);
            holder.thumbnail.setImageBitmap(null);
        }

        static class VH extends RecyclerView.ViewHolder {
            ImageView   thumbnail;
            TextView    riskLabel;
            TextView    regionCount;
            TextView    sourceUrl;
            Button      btnBlurNow;
            Button      btnOpenBlur;
            ProgressBar blurProgress;

            VH(View v) {
                super(v);
                thumbnail    = v.findViewById(R.id.ivThumbnail);
                riskLabel    = v.findViewById(R.id.tvRiskLabel);
                regionCount  = v.findViewById(R.id.tvRegionCount);
                sourceUrl    = v.findViewById(R.id.tvSourceUrl);
                btnBlurNow   = v.findViewById(R.id.btnBlurNow);
                btnOpenBlur  = v.findViewById(R.id.btnOpenBlur);
                blurProgress = v.findViewById(R.id.blurProgress);
            }
        }
    }
}
