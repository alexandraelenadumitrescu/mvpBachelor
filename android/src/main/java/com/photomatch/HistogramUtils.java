package com.photomatch;

import android.graphics.Bitmap;

/**
 * Computes normalised per-channel histograms from a Bitmap.
 * Returns float[3][256]: index 0=R, 1=G, 2=B, values in [0,1].
 */
public class HistogramUtils {

    private static final int SIZE = 256;

    /** Does not recycle the input bitmap. */
    public static float[][] compute(Bitmap bitmap) {
        Bitmap small = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true);
        int[] pixels = new int[SIZE * SIZE];
        small.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
        small.recycle();

        int[][] counts = new int[3][256];
        for (int px : pixels) {
            counts[0][(px >> 16) & 0xFF]++;
            counts[1][(px >>  8) & 0xFF]++;
            counts[2][ px        & 0xFF]++;
        }

        float total = pixels.length;
        float[][] hist = new float[3][256];
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 256; i++) {
                hist[c][i] = counts[c][i] / total;
            }
        }
        return hist;
    }
}
