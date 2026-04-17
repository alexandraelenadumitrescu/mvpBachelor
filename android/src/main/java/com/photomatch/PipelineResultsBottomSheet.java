package com.photomatch;

import android.content.ContentValues;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.google.android.material.bottomsheet.BottomSheetDialogFragment;
import com.photomatch.api.ApiClient;

import android.provider.MediaStore;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.concurrent.Executors;

public class PipelineResultsBottomSheet extends BottomSheetDialogFragment {

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_pipeline_results, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        List<PipelineResultsHolder.PipelinePhoto> photos =
            PipelineResultsHolder.instance.photos;

        // Summary text
        TextView tvSummary = view.findViewById(R.id.tvResultsSummary);
        int corrected = 0;
        for (PipelineResultsHolder.PipelinePhoto p : photos)
            if (p.correctedBase64 != null) corrected++;
        tvSummary.setText(photos.size() + " photos  \u2022  " + corrected + " corrected");

        // Grid
        RecyclerView rv = view.findViewById(R.id.rvResults);
        rv.setLayoutManager(new GridLayoutManager(requireContext(), 3));
        rv.setAdapter(new ResultsAdapter(photos));

        // Buttons
        view.findViewById(R.id.btnSaveAll).setOnClickListener(v -> saveAllToGallery(photos));
        view.findViewById(R.id.btnNewPipeline).setOnClickListener(v -> {
            dismiss();
            if (getActivity() instanceof PipelineActivity)
                ((PipelineActivity) getActivity()).resetPipeline();
        });
    }

    // ── Save to gallery ───────────────────────────────────────────────────────

    private void saveAllToGallery(List<PipelineResultsHolder.PipelinePhoto> photos) {
        Executors.newSingleThreadExecutor().execute(() -> {
            int saved = 0;
            for (int i = 0; i < photos.size(); i++) {
                PipelineResultsHolder.PipelinePhoto p = photos.get(i);
                if (p.correctedBase64 != null) {
                    Bitmap bmp = ApiClient.base64ToBitmap(p.correctedBase64);
                    if (bmp != null) {
                        MediaStore.Images.Media.insertImage(
                            requireContext().getContentResolver(),
                            bmp, "pipeline_" + i, "");
                        saved++;
                    }
                } else {
                    ContentValues cv = new ContentValues();
                    cv.put(MediaStore.Images.Media.DISPLAY_NAME, "pipeline_" + i + ".jpg");
                    cv.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
                    Uri dest = requireContext().getContentResolver()
                        .insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv);
                    if (dest != null) {
                        try (InputStream in = requireContext().getContentResolver()
                                .openInputStream(p.originalUri);
                             OutputStream out = requireContext().getContentResolver()
                                .openOutputStream(dest)) {
                            if (in != null && out != null) {
                                byte[] buf = new byte[8192];
                                int n;
                                while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
                                saved++;
                            }
                        } catch (IOException ignored) {}
                    }
                }
            }
            final int total = saved;
            requireActivity().runOnUiThread(() ->
                Toast.makeText(requireContext(), total + " photos saved", Toast.LENGTH_SHORT).show());
        });
    }

    // ── Inner adapter ─────────────────────────────────────────────────────────

    private static class ResultsAdapter
        extends RecyclerView.Adapter<ResultsAdapter.VH> {

        private final List<PipelineResultsHolder.PipelinePhoto> photos;

        ResultsAdapter(List<PipelineResultsHolder.PipelinePhoto> photos) {
            this.photos = photos;
        }

        @NonNull
        @Override
        public VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_pipeline_result, parent, false);
            return new VH(v);
        }

        @Override
        public void onBindViewHolder(@NonNull VH holder, int position) {
            PipelineResultsHolder.PipelinePhoto photo = photos.get(position);
            if (photo.correctedBase64 != null) {
                Bitmap bmp = ApiClient.base64ToBitmap(photo.correctedBase64);
                Glide.with(holder.ivThumb).load(bmp).into(holder.ivThumb);
                holder.tvTag.setText("CORRECTED");
            } else {
                Glide.with(holder.ivThumb).load(photo.originalUri).into(holder.ivThumb);
                holder.tvTag.setText("ORIGINAL");
            }
        }

        @Override
        public int getItemCount() { return photos.size(); }

        static class VH extends RecyclerView.ViewHolder {
            final ImageView ivThumb;
            final TextView  tvTag;
            VH(View v) {
                super(v);
                ivThumb = v.findViewById(R.id.ivResultThumb);
                tvTag   = v.findViewById(R.id.tvResultTag);
            }
        }
    }
}
