package com.photomatch.ui;

import android.content.Context;
import android.graphics.Matrix;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;

import androidx.appcompat.widget.AppCompatImageView;

/**
 * ImageView with pinch-to-zoom and drag.
 * Starts in fitCenter. Double-tap resets to fitCenter.
 * User must initiate all zoom — image never zooms automatically.
 */
public class ZoomableImageView extends AppCompatImageView {

    private final Matrix matrix = new Matrix();

    // scale relative to the fitCenter base (1.0 = fitCenter, 5.0 = 5x further)
    private float userScale = 1f;
    private static final float MAX_USER_SCALE = 5f;

    private final ScaleGestureDetector scaleDetector;
    private final GestureDetector      gestureDetector;

    private float lastX, lastY;
    private boolean dragging = false;

    public ZoomableImageView(Context ctx) { this(ctx, null); }

    public ZoomableImageView(Context ctx, AttributeSet attrs) {
        super(ctx, attrs);
        setScaleType(ScaleType.MATRIX);

        scaleDetector = new ScaleGestureDetector(ctx,
            new ScaleGestureDetector.SimpleOnScaleGestureListener() {
                @Override
                public boolean onScale(ScaleGestureDetector d) {
                    float factor   = d.getScaleFactor();
                    float newScale = userScale * factor;
                    newScale = Math.max(1f, Math.min(newScale, MAX_USER_SCALE));
                    float real = newScale / userScale;
                    userScale = newScale;
                    matrix.postScale(real, real, d.getFocusX(), d.getFocusY());
                    clamp();
                    setImageMatrix(matrix);
                    return true;
                }
            });

        gestureDetector = new GestureDetector(ctx,
            new GestureDetector.SimpleOnGestureListener() {
                @Override
                public boolean onDoubleTap(MotionEvent e) {
                    resetZoom();
                    return true;
                }
            });
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        gestureDetector.onTouchEvent(e);
        scaleDetector.onTouchEvent(e);

        switch (e.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                lastX = e.getX(); lastY = e.getY();
                dragging = true;
                break;
            case MotionEvent.ACTION_MOVE:
                if (dragging && !scaleDetector.isInProgress() && userScale > 1f) {
                    matrix.postTranslate(e.getX() - lastX, e.getY() - lastY);
                    clamp();
                    setImageMatrix(matrix);
                }
                lastX = e.getX(); lastY = e.getY();
                break;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                dragging = false;
                break;
        }
        return true;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /** Resets zoom to fitCenter. Call after setting a new image. */
    public void resetZoom() {
        userScale = 1f;
        applyFitCenter();
    }

    @Override
    public void setImageBitmap(android.graphics.Bitmap bm) {
        super.setImageBitmap(bm);
        // Wait for layout to know the view dimensions, then fit the image
        post(this::resetZoom);
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    /** Computes and applies a fitCenter matrix (image centered, scaled to fit). */
    private void applyFitCenter() {
        Drawable d = getDrawable();
        if (d == null || getWidth() == 0 || getHeight() == 0) return;

        float imgW = d.getIntrinsicWidth();
        float imgH = d.getIntrinsicHeight();
        if (imgW <= 0 || imgH <= 0) return;

        float viewW = getWidth();
        float viewH = getHeight();

        float fitScale = Math.min(viewW / imgW, viewH / imgH);
        float tx = (viewW - imgW * fitScale) / 2f;
        float ty = (viewH - imgH * fitScale) / 2f;

        matrix.reset();
        matrix.postScale(fitScale, fitScale);
        matrix.postTranslate(tx, ty);
        setImageMatrix(matrix);
    }

    /** Clamps translation so the image never leaves the view when zoomed. */
    private void clamp() {
        Drawable d = getDrawable();
        if (d == null) return;

        float[] v = new float[9];
        matrix.getValues(v);
        float curScale = v[Matrix.MSCALE_X];
        float tx       = v[Matrix.MTRANS_X];
        float ty       = v[Matrix.MTRANS_Y];

        float imgW = d.getIntrinsicWidth()  * curScale;
        float imgH = d.getIntrinsicHeight() * curScale;
        float viewW = getWidth();
        float viewH = getHeight();

        float dx = 0, dy = 0;
        if (imgW <= viewW) { dx = (viewW - imgW) / 2f - tx; }
        else {
            if (tx > 0)             dx = -tx;
            if (tx + imgW < viewW)  dx = viewW - tx - imgW;
        }
        if (imgH <= viewH) { dy = (viewH - imgH) / 2f - ty; }
        else {
            if (ty > 0)             dy = -ty;
            if (ty + imgH < viewH)  dy = viewH - ty - imgH;
        }
        if (dx != 0 || dy != 0) matrix.postTranslate(dx, dy);
    }
}
