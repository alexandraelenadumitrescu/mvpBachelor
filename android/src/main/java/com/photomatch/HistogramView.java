package com.photomatch;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

/**
 * Custom View that draws three overlapping RGB channel line charts.
 * R = red line, G = green line, B = blue line.
 * Dark #1A1A1A background, no fill under lines.
 */
public class HistogramView extends View {

    private static final int   BG_COLOR    = Color.parseColor("#1A1A1A");
    private static final int   COLOR_R     = Color.parseColor("#FF5555");
    private static final int   COLOR_G     = Color.parseColor("#55CC55");
    private static final int   COLOR_B     = Color.parseColor("#5599FF");
    private static final int   COLOR_LABEL = Color.parseColor("#C9A84C");
    private static final float LABEL_SP    = 11f;
    private static final float LINE_WIDTH  = 1.5f;

    private final Paint bgPaint    = new Paint();
    private final Paint linePaint  = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private float[][] hist;   // [3][256]
    private String    label;

    public HistogramView(Context ctx) { super(ctx); init(ctx); }
    public HistogramView(Context ctx, AttributeSet attrs) { super(ctx, attrs); init(ctx); }
    public HistogramView(Context ctx, AttributeSet attrs, int defStyle) {
        super(ctx, attrs, defStyle); init(ctx);
    }

    private void init(Context ctx) {
        bgPaint.setColor(BG_COLOR);
        bgPaint.setStyle(Paint.Style.FILL);

        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(LINE_WIDTH);
        linePaint.setStrokeCap(Paint.Cap.ROUND);
        linePaint.setStrokeJoin(Paint.Join.ROUND);

        labelPaint.setColor(COLOR_LABEL);
        labelPaint.setTextSize(LABEL_SP * ctx.getResources().getDisplayMetrics().scaledDensity);
    }

    /** Call from UI thread. */
    public void setData(float[][] hist, String label) {
        this.hist  = hist;
        this.label = label;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth();
        int h = getHeight();

        canvas.drawRect(0, 0, w, h, bgPaint);

        if (hist == null) return;

        float labelH = (label != null && !label.isEmpty())
            ? labelPaint.getTextSize() + 4f : 0f;

        if (label != null && !label.isEmpty()) {
            canvas.drawText(label, 4f, labelPaint.getTextSize(), labelPaint);
        }

        float chartTop    = labelH;
        float chartBottom = h - 2f;
        float chartH      = chartBottom - chartTop;

        // Find global max for y-scaling
        float maxVal = 0f;
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 256; i++) {
                if (hist[c][i] > maxVal) maxVal = hist[c][i];
            }
        }
        if (maxVal == 0f) return;

        int[]   colors  = { COLOR_R, COLOR_G, COLOR_B };
        float   xStep   = w / 256f;

        for (int c = 0; c < 3; c++) {
            linePaint.setColor(colors[c]);
            float prevX = 0f;
            float prevY = chartBottom - (hist[c][0] / maxVal) * chartH;
            for (int i = 1; i < 256; i++) {
                float x = i * xStep;
                float y = chartBottom - (hist[c][i] / maxVal) * chartH;
                canvas.drawLine(prevX, prevY, x, y, linePaint);
                prevX = x;
                prevY = y;
            }
        }
    }
}
