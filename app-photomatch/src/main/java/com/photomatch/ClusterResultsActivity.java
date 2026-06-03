package com.photomatch;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager2.widget.ViewPager2;

import com.bumptech.glide.Glide;
import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;
import com.google.gson.Gson;
import com.photomatch.api.ClusterInfo;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ClusterResultsActivity extends AppCompatActivity {

    private ClusterActivity.ClusterCache cache;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_cluster_results);

        String cachePath = getIntent().getStringExtra(ClusterActivity.EXTRA_CACHE_PATH);
        if (cachePath == null) { finish(); return; }

        try (FileReader fr = new FileReader(cachePath)) {
            cache = new Gson().fromJson(fr, ClusterActivity.ClusterCache.class);
        } catch (IOException e) {
            Toast.makeText(this, "Failed to load results", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        if (cache == null || cache.response == null || cache.response.clusters == null) {
            finish();
            return;
        }

        // Ensure clusters are in sequential clusterID order regardless of server response ordering
        cache.response.clusters.sort((a, b) -> Integer.compare(a.clusterID, b.clusterID));

        TextView tvScore   = findViewById(R.id.tvScore);
        TabLayout tabLayout = findViewById(R.id.tabLayout);
        ViewPager2 viewPager = findViewById(R.id.viewPager);

        tvScore.setText(String.format(Locale.US,
            "Similarity score: %.2f", cache.response.silhouetteScore));

        viewPager.setAdapter(new ClusterPagerAdapter());

        new TabLayoutMediator(tabLayout, viewPager, (tab, pos) -> {
            int size = cache.response.clusters.get(pos).indices.size();
            tab.setText("Group " + (pos + 1) + " (" + size + ")");
        }).attach();
    }

    // --- ViewPager2 adapter (one page per cluster) ---

    private class ClusterPagerAdapter extends RecyclerView.Adapter<ClusterPagerAdapter.PageHolder> {

        @Override
        public PageHolder onCreateViewHolder(ViewGroup parent, int viewType) {
            RecyclerView rv = new RecyclerView(parent.getContext());
            rv.setLayoutManager(new GridLayoutManager(parent.getContext(), 3));
            rv.setLayoutParams(new RecyclerView.LayoutParams(
                RecyclerView.LayoutParams.MATCH_PARENT,
                RecyclerView.LayoutParams.MATCH_PARENT));
            // Prevent ViewPager2 from intercepting vertical scroll events
            rv.setOnTouchListener((v, e) -> {
                v.getParent().requestDisallowInterceptTouchEvent(true);
                return false;
            });
            return new PageHolder(rv);
        }

        @Override
        public void onBindViewHolder(PageHolder holder, int position) {
            ((RecyclerView) holder.itemView).setAdapter(new ThumbnailGridAdapter(position));
        }

        @Override
        public int getItemCount() {
            return cache.response.clusters.size();
        }

        @Override
        public int getItemViewType(int position) {
            return position;
        }

        class PageHolder extends RecyclerView.ViewHolder {
            PageHolder(RecyclerView rv) { super(rv); }
        }
    }

    // --- Per-cluster thumbnail grid adapter ---

    private class ThumbnailGridAdapter extends RecyclerView.Adapter<ThumbnailGridAdapter.VH> {

        private final List<Uri> uris;

        ThumbnailGridAdapter(int clusterPosition) {
            ClusterInfo cluster = cache.response.clusters.get(clusterPosition);
            uris = new ArrayList<>(cluster.indices.size());
            for (int idx : cluster.indices) {
                if (cache.originalUris != null && idx < cache.originalUris.size()) {
                    uris.add(Uri.parse(cache.originalUris.get(idx)));
                }
            }
        }

        @Override
        public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            ImageView iv = (ImageView) getLayoutInflater()
                .inflate(R.layout.item_cluster_thumb, parent, false);
            return new VH(iv);
        }

        @Override
        public void onBindViewHolder(VH holder, int position) {
            Uri uri = uris.get(position);
            Glide.with(ClusterResultsActivity.this)
                .load(uri)
                .centerCrop()
                .into(holder.iv);
            holder.iv.setOnClickListener(v -> openUri(uri));
        }

        @Override public int getItemCount() { return uris.size(); }

        @Override
        public void onViewRecycled(VH holder) {
            super.onViewRecycled(holder);
            Glide.with(ClusterResultsActivity.this).clear(holder.iv);
        }

        class VH extends RecyclerView.ViewHolder {
            final ImageView iv;
            VH(ImageView iv) { super(iv); this.iv = iv; }
        }
    }

    private void openUri(Uri uri) {
        try {
            Intent intent = new Intent(Intent.ACTION_VIEW);
            intent.setDataAndType(uri, "image/*");
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(intent);
        } catch (Exception ignored) {}
    }
}
