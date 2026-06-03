package com.photomatch;

import android.graphics.Bitmap;

/**
 * Pure-Java aesthetic scorer. No ML model, no server call.
 * Computes a weighted sum of sharpness, exposure, contrast, and composition.
 */
public class AestheticScorer {

    private static final int   SIZE   = 256;
    private static final float SCORE_SHARPNESS_NORM  = 500f;
    private static final float SCORE_CONTRAST_NORM   = 60f;
    private static final float EXPOSURE_LOW           = 80f;
    private static final float EXPOSURE_HIGH          = 200f;

    /** Returns an aesthetic quality score in [0, 1]. Does not recycle the input bitmap. */
    public static float score(Bitmap bitmap) {
        Bitmap small = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true);
        int[] pixels = new int[SIZE * SIZE];
        small.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
        small.recycle();

        float[] lum = new float[SIZE * SIZE];
        for (int i = 0; i < pixels.length; i++) {
            int r = (pixels[i] >> 16) & 0xFF;
            int g = (pixels[i] >>  8) & 0xFF;
            int b =  pixels[i]        & 0xFF;
            lum[i] = 0.299f * r + 0.587f * g + 0.114f * b;
        }

        float sharpness   = computeSharpness(lum);
        float exposure    = computeExposure(lum);
        float contrast    = computeContrast(lum);
        float composition = computeComposition(lum);

        return 0.4f * sharpness + 0.3f * exposure + 0.2f * contrast + 0.1f * composition;
    }

    /** Laplacian variance — high = sharp. */
    private static float computeSharpness(float[] lum) {
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
        return Math.min(1f, variance / SCORE_SHARPNESS_NORM);
    }

    /** Mean luminance score — penalises under/over exposure. */
    private static float computeExposure(float[] lum) {
        float mean = 0f;
        for (float v : lum) mean += v;
        mean /= lum.length;
        if (mean >= EXPOSURE_LOW && mean <= EXPOSURE_HIGH) return 1f;
        if (mean < EXPOSURE_LOW)  return Math.max(0f, mean / EXPOSURE_LOW);
        return Math.max(0f, 1f - (mean - EXPOSURE_HIGH) / (255f - EXPOSURE_HIGH));
    }

    /** Standard deviation of luminance. */
    private static float computeContrast(float[] lum) {
        float mean = 0f;
        for (float v : lum) mean += v;
        mean /= lum.length;
        float var = 0f;
        for (float v : lum) { float d = v - mean; var += d * d; }
        float std = (float) Math.sqrt(var / lum.length);
        return Math.min(1f, std / SCORE_CONTRAST_NORM);
    }

    /**
     * Sobel gradient energy ratio: center third vs full image.
     * Higher centre energy = better composition / subject in frame.
     */
    private static float computeComposition(float[] lum) {
        int cx0 = SIZE / 3, cx1 = 2 * SIZE / 3;
        int cy0 = SIZE / 3, cy1 = 2 * SIZE / 3;
        float fullEnergy = 0f, centerEnergy = 0f;
        int   fullCount  = 0,  centerCount  = 0;
        for (int y = 1; y < SIZE - 1; y++) {
            for (int x = 1; x < SIZE - 1; x++) {
                float gx = lum[y * SIZE + (x + 1)] - lum[y * SIZE + (x - 1)];
                float gy = lum[(y + 1) * SIZE + x]  - lum[(y - 1) * SIZE + x];
                float mag = (float) Math.sqrt(gx * gx + gy * gy);
                fullEnergy += mag;
                fullCount++;
                if (x >= cx0 && x < cx1 && y >= cy0 && y < cy1) {
                    centerEnergy += mag;
                    centerCount++;
                }
            }
        }
        if (fullCount == 0 || fullEnergy < 1f) return 0.5f;
        float fullDensity   = fullEnergy   / fullCount;
        float centerDensity = centerEnergy / centerCount;
        return Math.min(1f, centerDensity / (fullDensity + 1e-6f));
    }
}
