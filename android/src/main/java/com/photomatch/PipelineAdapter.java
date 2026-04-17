package com.photomatch;

import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.photomatch.api.ApiClient;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class PipelineAdapter extends RecyclerView.Adapter<PipelineAdapter.VH> {

    interface OnDeleteListener {
        void onDelete(int position);
    }

    private final List<PipelineStep> steps;
    private OnDeleteListener deleteListener;

    // Track which positions have expanded panels
    private final Set<Integer> expandedDetails  = new HashSet<>();
    private final Set<Integer> expandedSettings = new HashSet<>();

    PipelineAdapter(List<PipelineStep> steps) {
        this.steps = steps;
    }

    void setOnDeleteListener(OnDeleteListener l) {
        this.deleteListener = l;
    }

    void moveItem(int from, int to) {
        Collections.swap(steps, from, to);
        notifyItemMoved(from, to);
    }

    @NonNull
    @Override
    public VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext())
            .inflate(R.layout.item_pipeline_step, parent, false);
        return new VH(v);
    }

    @Override
    public void onBindViewHolder(@NonNull VH holder, int position) {
        PipelineStep step = steps.get(position);

        // Amber circle background for step number
        GradientDrawable circle = new GradientDrawable();
        circle.setShape(GradientDrawable.OVAL);
        circle.setColor(Color.parseColor("#C9A84C"));
        holder.tvStepNum.setBackground(circle);
        holder.tvStepNum.setText(String.valueOf(position + 1));

        holder.tvStepName.setText(step.name());
        holder.tvStepSubtitle.setText(step.subtitle());

        // Connector line visible for all but last item
        holder.connector.setVisibility(
            position < getItemCount() - 1 ? View.VISIBLE : View.GONE);

        // Delete button: only shown when step is IDLE
        boolean idle = step.status == PipelineStep.Status.IDLE;
        holder.btnDelete.setVisibility(idle ? View.VISIBLE : View.INVISIBLE);
        holder.btnDelete.setOnClickListener(v -> {
            int pos = holder.getAdapterPosition();
            if (pos != RecyclerView.NO_POSITION && deleteListener != null) {
                expandedDetails.remove(pos);
                expandedSettings.remove(pos);
                deleteListener.onDelete(pos);
            }
        });

        // Settings gear button
        holder.btnSettings.setOnClickListener(v -> {
            int pos = holder.getAdapterPosition();
            if (pos == RecyclerView.NO_POSITION) return;
            if (expandedSettings.contains(pos)) {
                expandedSettings.remove(pos);
                animateCollapse(holder.settingsPanel);
            } else {
                expandedSettings.add(pos);
                animateExpand(holder.settingsPanel);
            }
        });

        // Card body tap (not gear/delete) → toggle details panel
        holder.itemView.setOnClickListener(v -> {
            int pos = holder.getAdapterPosition();
            if (pos == RecyclerView.NO_POSITION) return;
            if (expandedDetails.contains(pos)) {
                expandedDetails.remove(pos);
                animateCollapse(holder.detailsPanel);
            } else {
                expandedDetails.add(pos);
                bindDetailsPanel(holder, step);
                animateExpand(holder.detailsPanel);
            }
        });

        // Status indicator
        switch (step.status) {
            case IDLE:
                holder.pbStep.setVisibility(View.GONE);
                holder.tvStepResult.setVisibility(View.GONE);
                break;
            case RUNNING:
                holder.pbStep.setVisibility(View.VISIBLE);
                holder.tvStepResult.setVisibility(View.GONE);
                break;
            case DONE:
                holder.pbStep.setVisibility(View.GONE);
                holder.tvStepResult.setVisibility(View.VISIBLE);
                holder.tvStepResult.setTextColor(Color.parseColor("#55CC55"));
                holder.tvStepResult.setText("\u2713 " + step.summary);
                break;
            case ERROR:
                holder.pbStep.setVisibility(View.GONE);
                holder.tvStepResult.setVisibility(View.VISIBLE);
                holder.tvStepResult.setTextColor(Color.parseColor("#FF5555"));
                holder.tvStepResult.setText(step.summary);
                break;
        }

        // Restore panel visibility without animation on rebind
        holder.settingsPanel.setVisibility(expandedSettings.contains(position) ? View.VISIBLE : View.GONE);
        holder.detailsPanel.setVisibility(expandedDetails.contains(position) ? View.VISIBLE : View.GONE);

        // Always bind settings panel so SeekBar reflects current strengthParam
        bindSettingsPanel(holder, step);

        // If detail panel is expanded, populate content
        if (expandedDetails.contains(position)) {
            bindDetailsPanel(holder, step);
        }
    }

    @Override
    public int getItemCount() {
        return steps.size();
    }

    // ── Details panel ─────────────────────────────────────────────────────────

    private void bindDetailsPanel(VH holder, PipelineStep step) {
        switch (step.status) {
            case IDLE:
                holder.tvDetailLine.setText("Step not yet run");
                holder.thumbScroll.setVisibility(View.GONE);
                break;
            case RUNNING:
                holder.tvDetailLine.setText("Running\u2026");
                holder.thumbScroll.setVisibility(View.GONE);
                break;
            case ERROR:
                holder.tvDetailLine.setText(step.summary);
                holder.thumbScroll.setVisibility(View.GONE);
                break;
            case DONE:
                holder.tvDetailLine.setText(
                    step.resultDetail.isEmpty() ? step.summary : step.resultDetail);

                boolean hasUris   = !step.resultUris.isEmpty();
                boolean hasBase64 = !step.resultBase64.isEmpty();
                if (hasUris || hasBase64) {
                    holder.thumbScroll.setVisibility(View.VISIBLE);
                    populateThumbs(holder, step);
                } else {
                    holder.thumbScroll.setVisibility(View.GONE);
                }
                break;
        }
    }

    private void populateThumbs(VH holder, PipelineStep step) {
        Context ctx = holder.itemView.getContext();
        holder.thumbRow.removeAllViews();

        int sizePx = (int) (72 * ctx.getResources().getDisplayMetrics().density);
        int marginPx = (int) (4 * ctx.getResources().getDisplayMetrics().density);

        // Show up to 5 thumbnails, preferring resultBase64 over resultUris
        int count = Math.min(5, Math.max(step.resultUris.size(), step.resultBase64.size()));
        for (int i = 0; i < count; i++) {
            ImageView iv = new ImageView(ctx);
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(sizePx, sizePx);
            lp.setMarginEnd(marginPx);
            iv.setLayoutParams(lp);
            iv.setScaleType(ImageView.ScaleType.CENTER_CROP);

            if (i < step.resultBase64.size() && step.resultBase64.get(i) != null) {
                Bitmap bmp = ApiClient.base64ToBitmap(step.resultBase64.get(i));
                Glide.with(ctx).load(bmp).into(iv);
            } else if (i < step.resultUris.size()) {
                Glide.with(ctx).load(step.resultUris.get(i)).into(iv);
            }
            holder.thumbRow.addView(iv);
        }
    }

    // ── Settings panel ────────────────────────────────────────────────────────

    private void bindSettingsPanel(VH holder, PipelineStep step) {
        holder.tvSliderLabel.setText(step.sliderLabel());
        holder.tvSliderMin.setText(step.sliderMinLabel());
        holder.tvSliderMax.setText(step.sliderMaxLabel());

        holder.seekStrength.setMax(step.sliderMax());
        holder.seekStrength.setProgress(step.sliderProgress());

        holder.seekStrength.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) step.strengthParam = step.progressToParam(progress);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
    }

    // ── Expand / collapse animation ───────────────────────────────────────────

    static void animateExpand(View v) {
        v.measure(
            View.MeasureSpec.makeMeasureSpec(
                ((View) v.getParent()).getWidth(), View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED));
        int target = v.getMeasuredHeight();
        v.getLayoutParams().height = 0;
        v.setVisibility(View.VISIBLE);
        ValueAnimator anim = ValueAnimator.ofInt(0, target);
        anim.setDuration(200);
        anim.addUpdateListener(a -> {
            v.getLayoutParams().height = (int) a.getAnimatedValue();
            v.requestLayout();
        });
        anim.start();
    }

    static void animateCollapse(View v) {
        int initial = v.getMeasuredHeight();
        ValueAnimator anim = ValueAnimator.ofInt(initial, 0);
        anim.setDuration(200);
        anim.addUpdateListener(a -> {
            v.getLayoutParams().height = (int) a.getAnimatedValue();
            v.requestLayout();
        });
        anim.addListener(new android.animation.AnimatorListenerAdapter() {
            @Override public void onAnimationEnd(android.animation.Animator animation) {
                v.setVisibility(View.GONE);
                v.getLayoutParams().height = ViewGroup.LayoutParams.WRAP_CONTENT;
            }
        });
        anim.start();
    }

    // ── ViewHolder ────────────────────────────────────────────────────────────

    static class VH extends RecyclerView.ViewHolder {
        final TextView     tvStepNum;
        final TextView     tvStepName;
        final TextView     tvStepSubtitle;
        final ProgressBar  pbStep;
        final TextView     tvStepResult;
        final Button       btnSettings;
        final Button       btnDelete;
        final View         connector;

        // Settings panel
        final LinearLayout settingsPanel;
        final TextView     tvSliderLabel;
        final SeekBar      seekStrength;
        final TextView     tvSliderMin;
        final TextView     tvSliderMax;

        // Details panel
        final LinearLayout detailsPanel;
        final TextView     tvDetailLine;
        final View         thumbScroll;
        final LinearLayout thumbRow;

        VH(View v) {
            super(v);
            tvStepNum      = v.findViewById(R.id.tvStepNum);
            tvStepName     = v.findViewById(R.id.tvStepName);
            tvStepSubtitle = v.findViewById(R.id.tvStepSubtitle);
            pbStep         = v.findViewById(R.id.pbStep);
            tvStepResult   = v.findViewById(R.id.tvStepResult);
            btnSettings    = v.findViewById(R.id.btnSettings);
            btnDelete      = v.findViewById(R.id.btnDelete);
            connector      = v.findViewById(R.id.connector);

            settingsPanel  = v.findViewById(R.id.settingsPanel);
            tvSliderLabel  = v.findViewById(R.id.tvSliderLabel);
            seekStrength   = v.findViewById(R.id.seekStrength);
            tvSliderMin    = v.findViewById(R.id.tvSliderMin);
            tvSliderMax    = v.findViewById(R.id.tvSliderMax);

            detailsPanel   = v.findViewById(R.id.detailsPanel);
            tvDetailLine   = v.findViewById(R.id.tvDetailLine);
            thumbScroll    = v.findViewById(R.id.thumbScroll);
            thumbRow       = v.findViewById(R.id.thumbRow);
        }
    }
}
