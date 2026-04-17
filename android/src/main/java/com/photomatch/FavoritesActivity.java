package com.photomatch;

import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.photomatch.db.AppDatabase;
import com.photomatch.db.FavoritePhoto;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executors;

public class FavoritesActivity extends AppCompatActivity {

    static FavoritePhoto selectedFavorite;

    private RecyclerView rvFavorites;
    private TextView     tvEmpty;
    private FavoritesAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_favorites);

        tvEmpty      = findViewById(R.id.tvEmpty);
        rvFavorites  = findViewById(R.id.rvFavorites);
        rvFavorites.setLayoutManager(new GridLayoutManager(this, 2));
        adapter = new FavoritesAdapter(new ArrayList<>());
        rvFavorites.setAdapter(adapter);
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadFavorites();
    }

    private void loadFavorites() {
        Executors.newSingleThreadExecutor().execute(() -> {
            List<FavoritePhoto> items = AppDatabase.get(this).favoriteDao().getAll();
            runOnUiThread(() -> {
                adapter.setItems(items);
                tvEmpty.setVisibility(items.isEmpty() ? View.VISIBLE : View.GONE);
            });
        });
    }

    private class FavoritesAdapter extends RecyclerView.Adapter<FavoritesAdapter.VH> {

        private List<FavoritePhoto> items;

        FavoritesAdapter(List<FavoritePhoto> items) {
            this.items = items;
        }

        void setItems(List<FavoritePhoto> items) {
            this.items = items;
            notifyDataSetChanged();
        }

        @Override
        public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            android.view.View view = getLayoutInflater()
                .inflate(R.layout.item_favorite, parent, false);
            return new VH(view);
        }

        @Override
        public void onBindViewHolder(VH holder, int position) {
            FavoritePhoto item = items.get(position);

            if (item.uriString != null) {
                Glide.with(FavoritesActivity.this)
                    .load(Uri.parse(item.uriString))
                    .placeholder(new ColorDrawable(Color.parseColor("#1A1A1A")))
                    .centerCrop()
                    .into(holder.ivCorrected);
            } else {
                byte[] bytes = decodeBase64(item.correctedBase64);
                Glide.with(FavoritesActivity.this)
                    .load(bytes)
                    .placeholder(new ColorDrawable(Color.parseColor("#1A1A1A")))
                    .centerCrop()
                    .into(holder.ivCorrected);
            }

            SimpleDateFormat sdf = new SimpleDateFormat("MMM d", Locale.US);
            holder.tvTimestamp.setText(sdf.format(new Date(item.timestamp)));

            holder.itemView.setOnClickListener(v -> {
                selectedFavorite = item;
                startActivity(new Intent(FavoritesActivity.this, FavoriteDetailActivity.class));
            });
        }

        @Override public int getItemCount() { return items.size(); }

        @Override
        public void onViewRecycled(VH holder) {
            super.onViewRecycled(holder);
            Glide.with(FavoritesActivity.this).clear(holder.ivCorrected);
        }

        class VH extends RecyclerView.ViewHolder {
            final ImageView ivCorrected;
            final TextView  tvTimestamp;
            VH(android.view.View v) {
                super(v);
                ivCorrected = v.findViewById(R.id.ivCorrected);
                tvTimestamp = v.findViewById(R.id.tvTimestamp);
            }
        }
    }

    private static byte[] decodeBase64(String b64) {
        if (b64 == null) return null;
        return android.util.Base64.decode(b64, android.util.Base64.DEFAULT);
    }
}
