package com.photomatch;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
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

import com.photomatch.api.ScrapeRequest;
import com.photomatch.api.ScrapeResponse;
import com.photomatch.api.ScrapedImageResult;
import com.photomatch.base.BaseServerActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class WebScraperActivity extends BaseServerActivity {

    private EditText    etUrl;
    private Button      btnScrape;
    private RecyclerView rvResults;
    private TextView    tvStatus;
    private TextView    tvError;
    private ProgressBar progressBar;

    private final List<ScrapedImageResult> displayedResults = new ArrayList<>();
    private ScrapeResultAdapter adapter;

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

        adapter = new ScrapeResultAdapter(displayedResults, this::openInBlurEditor);
        rvResults.setLayoutManager(new LinearLayoutManager(this));
        rvResults.setAdapter(adapter);

        btnScrape.setOnClickListener(v -> startScrape());
    }

    private void startScrape() {
        String url = etUrl.getText().toString().trim();
        if (url.isEmpty()) {
            showError("Introdu un URL");
            return;
        }
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            showError("URL-ul trebuie sa inceapa cu http:// sau https://");
            return;
        }

        setProcessing(true);
        tvStatus.setVisibility(View.GONE);
        tvError.setVisibility(View.GONE);
        displayedResults.clear();
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
                List<ScrapedImageResult> risky = new ArrayList<>();
                if (body.results != null) {
                    for (ScrapedImageResult r : body.results) {
                        if (r.riskScore > 0) risky.add(r);
                    }
                }
                displayedResults.clear();
                displayedResults.addAll(risky);
                adapter.notifyDataSetChanged();

                tvStatus.setText(String.format(Locale.US,
                    "%d imagini riscante din %d procesate (%d sarite)",
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

    private void openInBlurEditor(ScrapedImageResult item, int position) {
        try {
            byte[] bytes = Base64.decode(item.thumbnailB64, Base64.DEFAULT);
            File tmpFile = new File(getCacheDir(), "scrape_thumb_" + position + ".jpg");
            try (FileOutputStream fos = new FileOutputStream(tmpFile)) {
                fos.write(bytes);
            }
            Uri uri = FileProvider.getUriForFile(this, "com.photomatch.fileprovider", tmpFile);
            ArrayList<Uri> uris = new ArrayList<>();
            uris.add(uri);
            Intent intent = new Intent(this, BlurActivity.class);
            intent.putParcelableArrayListExtra(BlurActivity.EXTRA_URIS, uris);
            startActivity(intent);
        } catch (Exception e) {
            showError("Nu se poate deschide: " + e.getMessage());
        }
    }

    // ── Adapter ───────────────────────────────────────────────────────────────

    interface OnItemAction {
        void onOpenBlurEditor(ScrapedImageResult item, int position);
    }

    static class ScrapeResultAdapter
            extends RecyclerView.Adapter<ScrapeResultAdapter.VH> {

        private final List<ScrapedImageResult> items;
        private final OnItemAction             listener;

        ScrapeResultAdapter(List<ScrapedImageResult> items, OnItemAction listener) {
            this.items    = items;
            this.listener = listener;
        }

        @Override
        public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            View v = android.view.LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_scrape_result, parent, false);
            return new VH(v);
        }

        @Override
        public void onBindViewHolder(VH holder, int position) {
            ScrapedImageResult item = items.get(position);

            // Decode thumbnail
            try {
                byte[] bytes = Base64.decode(item.thumbnailB64, Base64.DEFAULT);
                Bitmap bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
                holder.thumbnail.setImageBitmap(bmp);
            } catch (Exception ignored) {
                holder.thumbnail.setImageResource(android.R.drawable.ic_menu_gallery);
            }

            // Risk label with color
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

            holder.btnOpenBlur.setOnClickListener(v ->
                listener.onOpenBlurEditor(item, holder.getAdapterPosition()));
        }

        @Override
        public int getItemCount() { return items.size(); }

        @Override
        public void onViewRecycled(VH holder) {
            super.onViewRecycled(holder);
            holder.thumbnail.setImageBitmap(null);
        }

        static class VH extends RecyclerView.ViewHolder {
            ImageView thumbnail;
            TextView  riskLabel;
            TextView  regionCount;
            TextView  sourceUrl;
            Button    btnOpenBlur;

            VH(View v) {
                super(v);
                thumbnail   = v.findViewById(R.id.ivThumbnail);
                riskLabel   = v.findViewById(R.id.tvRiskLabel);
                regionCount = v.findViewById(R.id.tvRegionCount);
                sourceUrl   = v.findViewById(R.id.tvSourceUrl);
                btnOpenBlur = v.findViewById(R.id.btnOpenBlur);
            }
        }
    }
}
