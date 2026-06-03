package com.photomatch.ui;

import android.content.ContentResolver;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.recyclerview.widget.RecyclerView;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;

public class ThumbnailAdapter extends RecyclerView.Adapter<ThumbnailAdapter.VH> {

    private final ContentResolver contentResolver;
    private List<Uri> uris = new ArrayList<>();

    public ThumbnailAdapter(ContentResolver contentResolver) {
        this.contentResolver = contentResolver;
    }

    public void setUris(List<Uri> uris) {
        this.uris = uris;
        notifyDataSetChanged();
    }

    @Override
    public VH onCreateViewHolder(ViewGroup parent, int viewType) {
        ImageView iv = new ImageView(parent.getContext());
        float density = parent.getResources().getDisplayMetrics().density;
        int sz = (int) (80 * density);
        int mg = (int) (2  * density);
        RecyclerView.LayoutParams lp = new RecyclerView.LayoutParams(sz, sz);
        lp.setMargins(mg, mg, mg, mg);
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
                Bitmap bmp;
                try (InputStream is = contentResolver.openInputStream(uri)) {
                    bmp = BitmapFactory.decodeStream(is, null, opts);
                }
                if (bmp != null) holder.iv.post(() -> holder.iv.setImageBitmap(bmp));
            } catch (IOException ignored) {}
        });
    }

    @Override
    public int getItemCount() { return uris.size(); }

    public static class VH extends RecyclerView.ViewHolder {
        final ImageView iv;
        public VH(ImageView iv) { super(iv); this.iv = iv; }
    }
}
