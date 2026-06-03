package com.photomatch;

import android.graphics.Bitmap;

/**
 * On-device blur detector using Laplacian variance.
 * Same algorithm as AestheticScorer.computeSharpness() but returns
 * the raw variance score (not normalised to [0,1]) for a direct threshold check.
 */
public class BlurDetector {

    static final float BLUR_THRESHOLD = 100.0f;

    private static final int SIZE = 256;

    public static class BlurResult {
        public final float   score;
        public final boolean isBlurry;
        BlurResult(float score, float threshold) {
            this.score    = score;
            this.isBlurry = score < threshold;
        }
    }

    /** Does not recycle the input bitmap. Uses default threshold. */
    public static BlurResult check(Bitmap bitmap) {
        return check(bitmap, BLUR_THRESHOLD);
    }

    /** Does not recycle the input bitmap. Uses the provided threshold. */
    public static BlurResult check(Bitmap bitmap, float threshold) {
        Bitmap small = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true);
        int[] pixels = new int[SIZE * SIZE];
        small.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
        small.recycle();

        // Grayscale luminance
        float[] lum = new float[SIZE * SIZE];
        for (int i = 0; i < pixels.length; i++) {
            int r = (pixels[i] >> 16) & 0xFF;
            int g = (pixels[i] >>  8) & 0xFF;
            int b =  pixels[i]        & 0xFF;
            lum[i] = 0.299f * r + 0.587f * g + 0.114f * b;
        }

        // 4-neighbour Laplacian variance
        float sum = 0f, sumSq = 0f;
        int count = 0;
        for (int y = 1; y < SIZE - 1; y++) {
            for (int x = 1; x < SIZE - 1; x++) {
                float lap = 4f * lum[y * SIZE + x]
                        - lum[(y - 1) * SIZE + x]
                        - lum[(y + 1) * SIZE + x]
                        - lum[y * SIZE + (x - 1)]
                        - lum[y * SIZE + (x + 1)];
                sum   += lap;
                sumSq += lap * lap;
                count++;
            }
        }
        float mean     = sum / count;
        float variance = sumSq / count - mean * mean;
        return new BlurResult(variance, threshold);
    }
}
