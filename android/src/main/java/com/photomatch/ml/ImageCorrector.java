package com.photomatch.ml;

import android.graphics.Bitmap;

/**
 * On-device colour correction — mirrors the server's correct_clahe() + apply_lut_moderated().
 *
 * Pipeline (identical to server):
 *   1. RGB → CIE L*a*b*
 *   2. CLAHE on L channel (4×4 tiles, clipLimit = 1.5) — matches cv2.createCLAHE
 *   3. L*a*b* → RGB
 *   4. Trilinear 3-D LUT interpolation (17³ grid)
 *   5. Blend: result = clahe*(1-0.2) + lut_result*0.2
 *
 * If {@code lut} is null the method returns the CLAHE-corrected bitmap (step 3 only).
 */
public class ImageCorrector {

    public static final float LUT_STRENGTH = 0.2f;

    // D65 reference white
    private static final float Xn = 0.95047f;
    private static final float Yn = 1.00000f;
    private static final float Zn = 1.08883f;

    // sRGB → XYZ (D65) matrix (row-major)
    private static final float[] M_RGB2XYZ = {
        0.4124564f, 0.3575761f, 0.1804375f,
        0.2126729f, 0.7151522f, 0.0721750f,
        0.0193339f, 0.1191920f, 0.9503041f
    };

    // XYZ → sRGB (D65) matrix (row-major)
    private static final float[] M_XYZ2RGB = {
         3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f,  1.8760108f,  0.0415560f,
         0.0556434f, -0.2040259f,  1.0572252f
    };

    // CLAHE params — match server: clipLimit=1.5, tileGridSize=(4,4)
    private static final int   TILE_ROWS   = 4;
    private static final int   TILE_COLS   = 4;
    private static final float CLIP_LIMIT  = 1.5f;
    private static final int   HIST_BINS   = 256;

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Apply CLAHE + optional LUT correction.
     * @param bmp  Input bitmap (any config; not mutated).
     * @param lut  Flat float32 LUT from {@link LutCache}, or null for CLAHE-only.
     * @return     New ARGB_8888 bitmap with correction applied.
     */
    public static Bitmap correct(Bitmap bmp, float[] lut) {
        int w = bmp.getWidth();
        int h = bmp.getHeight();

        int[] pixels = new int[w * h];
        bmp.getPixels(pixels, 0, w, 0, 0, w, h);

        // --- Step 1-3: CLAHE in L*a*b* space ---
        float[] lab = rgbPixelsToLab(pixels);          // L in [0,100], a,b in [-128,127]
        applyClaheLab(lab, w, h);                      // modifies L in-place
        int[] clahePixels = labToRgbPixels(lab, pixels); // re-use alpha from original

        if (lut == null) {
            Bitmap result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
            result.setPixels(clahePixels, 0, w, 0, 0, w, h);
            return result;
        }

        // --- Step 4: trilinear LUT ---
        int lutSize = Math.round((float) Math.cbrt(lut.length / 3.0));
        int[] lutPixels = applyLut(clahePixels, lut, lutSize);

        // --- Step 5: blend clahe*(1-s) + lut*s ---
        int[] blended = blend(clahePixels, lutPixels, LUT_STRENGTH);

        Bitmap result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        result.setPixels(blended, 0, w, 0, 0, w, h);
        return result;
    }

    // ── Colour-space helpers ─────────────────────────────────────────────────

    /** Convert ARGB pixel array → flat L*a*b* array [L0,a0,b0, L1,a1,b1, …]. */
    private static float[] rgbPixelsToLab(int[] pixels) {
        float[] lab = new float[pixels.length * 3];
        for (int i = 0; i < pixels.length; i++) {
            int p = pixels[i];
            float r = linearize(((p >> 16) & 0xFF) / 255f);
            float g = linearize(((p >>  8) & 0xFF) / 255f);
            float b = linearize(( p        & 0xFF) / 255f);

            float X = M_RGB2XYZ[0]*r + M_RGB2XYZ[1]*g + M_RGB2XYZ[2]*b;
            float Y = M_RGB2XYZ[3]*r + M_RGB2XYZ[4]*g + M_RGB2XYZ[5]*b;
            float Z = M_RGB2XYZ[6]*r + M_RGB2XYZ[7]*g + M_RGB2XYZ[8]*b;

            float fx = labF(X / Xn);
            float fy = labF(Y / Yn);
            float fz = labF(Z / Zn);

            lab[i*3    ] = 116f * fy - 16f;          // L* [0,100]
            lab[i*3 + 1] = 500f * (fx - fy);          // a* [-128,127]
            lab[i*3 + 2] = 200f * (fy - fz);          // b*
        }
        return lab;
    }

    /** Convert flat L*a*b* back to ARGB pixel array, preserving original alpha. */
    private static int[] labToRgbPixels(float[] lab, int[] originalPixels) {
        int[] pixels = new int[originalPixels.length];
        for (int i = 0; i < pixels.length; i++) {
            float L = lab[i*3    ];
            float a = lab[i*3 + 1];
            float b = lab[i*3 + 2];

            float fy = (L + 16f) / 116f;
            float fx = a / 500f + fy;
            float fz = fy - b / 200f;

            float X = Xn * labFInv(fx);
            float Y = Yn * labFInv(fy);
            float Z = Zn * labFInv(fz);

            float r = M_XYZ2RGB[0]*X + M_XYZ2RGB[1]*Y + M_XYZ2RGB[2]*Z;
            float g = M_XYZ2RGB[3]*X + M_XYZ2RGB[4]*Y + M_XYZ2RGB[5]*Z;
            float bv = M_XYZ2RGB[6]*X + M_XYZ2RGB[7]*Y + M_XYZ2RGB[8]*Z;

            int ri = clamp255(Math.round(delinearize(r) * 255f));
            int gi = clamp255(Math.round(delinearize(g) * 255f));
            int bi = clamp255(Math.round(delinearize(bv) * 255f));
            int alpha = originalPixels[i] & 0xFF000000;
            pixels[i] = alpha | (ri << 16) | (gi << 8) | bi;
        }
        return pixels;
    }

    private static float linearize(float c) {
        return c <= 0.04045f ? c / 12.92f : (float) Math.pow((c + 0.055f) / 1.055f, 2.4);
    }

    private static float delinearize(float c) {
        c = Math.max(0f, Math.min(1f, c));
        return c <= 0.0031308f ? 12.92f * c : 1.055f * (float) Math.pow(c, 1.0 / 2.4) - 0.055f;
    }

    private static float labF(float t) {
        final float delta = 6f / 29f;
        return t > delta * delta * delta
            ? (float) Math.cbrt(t)
            : t / (3f * delta * delta) + 4f / 29f;
    }

    private static float labFInv(float t) {
        final float delta = 6f / 29f;
        return t > delta ? t * t * t : 3f * delta * delta * (t - 4f / 29f);
    }

    // ── CLAHE ─────────────────────────────────────────────────────────────────

    /**
     * Apply CLAHE on the L channel in-place.
     * L is in [0,100]. Internally scaled to [0,255] to match OpenCV's behaviour.
     */
    private static void applyClaheLab(float[] lab, int w, int h) {
        // Scale L to [0,255]
        int n = w * h;
        int[] lScaled = new int[n];
        for (int i = 0; i < n; i++) {
            lScaled[i] = clamp255(Math.round(lab[i*3] * 2.55f));
        }

        int[] equalized = clahe(lScaled, w, h,
            TILE_ROWS, TILE_COLS, CLIP_LIMIT, HIST_BINS);

        // Write back to lab array
        for (int i = 0; i < n; i++) {
            lab[i*3] = equalized[i] / 2.55f;   // back to [0,100]
        }
    }

    /**
     * Full CLAHE implementation.
     * Returns equalized values (same int array, [0, histBins-1] range).
     */
    private static int[] clahe(int[] channel, int w, int h,
                                int tileRows, int tileCols,
                                float clipLimit, int histBins) {
        int tileH = h / tileRows;
        int tileW = w / tileCols;
        if (tileH < 1) tileH = 1;
        if (tileW < 1) tileW = 1;

        // Build one CDF table per tile
        // cdfs[row][col][bin] = equalized value for that bin
        int[][][] cdfs = new int[tileRows][tileCols][histBins];

        for (int tr = 0; tr < tileRows; tr++) {
            for (int tc = 0; tc < tileCols; tc++) {
                int y0 = tr * tileH;
                int x0 = tc * tileW;
                int y1 = (tr == tileRows - 1) ? h : y0 + tileH;
                int x1 = (tc == tileCols - 1) ? w : x0 + tileW;

                // histogram
                int[] hist = new int[histBins];
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                        hist[channel[y * w + x] * (histBins - 1) / 255]++;

                // clip & redistribute
                clipHistogram(hist, clipLimit, (y1 - y0) * (x1 - x0));

                // CDF → equalization map
                int pixels = (y1 - y0) * (x1 - x0);
                buildCdf(hist, cdfs[tr][tc], histBins, pixels);
            }
        }

        // Bilinear interpolation of the 4 surrounding tile CDFs
        int[] out = new int[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int bin = channel[y * w + x] * (histBins - 1) / 255;

                // Tile-centre coordinates — clamp first so edge pixels replicate
                // the nearest tile CDF (matches OpenCV CLAHE border behaviour).
                float ty = Math.max(0f, Math.min(tileRows - 1f,
                               (y - tileH / 2f) / tileH));
                float tx = Math.max(0f, Math.min(tileCols - 1f,
                               (x - tileW / 2f) / tileW));

                int r0 = (int) ty;
                int c0 = (int) tx;
                float wr = ty - r0;
                float wc = tx - c0;

                int r1 = Math.min(tileRows - 1, r0 + 1);
                int c1 = Math.min(tileCols - 1, c0 + 1);

                float v00 = cdfs[r0][c0][bin];
                float v01 = cdfs[r0][c1][bin];
                float v10 = cdfs[r1][c0][bin];
                float v11 = cdfs[r1][c1][bin];

                float interp = (1 - wr) * ((1 - wc) * v00 + wc * v01)
                                  + wr  * ((1 - wc) * v10 + wc * v11);
                out[y * w + x] = clamp255(Math.round(interp));
            }
        }
        return out;
    }

    /** Clip histogram at clipLimit × average and redistribute excess uniformly. */
    private static void clipHistogram(int[] hist, float clipLimit, int pixelCount) {
        int clip = Math.max(1, Math.round(clipLimit * pixelCount / hist.length));
        int excess = 0;
        for (int b = 0; b < hist.length; b++) {
            if (hist[b] > clip) {
                excess += hist[b] - clip;
                hist[b] = clip;
            }
        }
        int addPerBin = excess / hist.length;
        int remainder = excess - addPerBin * hist.length;
        for (int b = 0; b < hist.length; b++) {
            hist[b] += addPerBin;
            if (remainder > 0) { hist[b]++; remainder--; }
        }
    }

    /** Build equalization map from clipped histogram CDF. */
    private static void buildCdf(int[] hist, int[] cdf, int histBins, int pixelCount) {
        int cumSum = 0;
        int cdfMin = -1;
        for (int b = 0; b < histBins; b++) {
            cumSum += hist[b];
            if (cdfMin < 0 && hist[b] > 0) cdfMin = cumSum - hist[b];
            cdf[b] = (int) Math.round(
                (cumSum - cdfMin) / (float) Math.max(1, pixelCount - cdfMin) * 255f);
        }
    }

    // ── 3-D LUT (trilinear interpolation) ────────────────────────────────────

    /**
     * Apply a flat float32 3-D LUT to a pixel array.
     * LUT layout: [r][g][b][c] with C order — index = (ri*N*N + gi*N + bi)*3 + c
     */
    private static int[] applyLut(int[] pixels, float[] lut, int N) {
        int[] out = new int[pixels.length];
        float scale = (N - 1f);
        for (int i = 0; i < pixels.length; i++) {
            int p = pixels[i];
            float r = ((p >> 16) & 0xFF) / 255f;
            float g = ((p >>  8) & 0xFF) / 255f;
            float b = ( p        & 0xFF) / 255f;

            float ri = r * scale, gi = g * scale, bi = b * scale;
            int r0 = (int) ri, g0 = (int) gi, b0 = (int) bi;
            int r1 = Math.min(r0 + 1, N - 1);
            int g1 = Math.min(g0 + 1, N - 1);
            int b1 = Math.min(b0 + 1, N - 1);
            float dr = ri - r0, dg = gi - g0, db = bi - b0;

            float[] result = new float[3];
            for (int c = 0; c < 3; c++) {
                float v000 = lut[(r0*N*N + g0*N + b0)*3 + c];
                float v001 = lut[(r0*N*N + g0*N + b1)*3 + c];
                float v010 = lut[(r0*N*N + g1*N + b0)*3 + c];
                float v011 = lut[(r0*N*N + g1*N + b1)*3 + c];
                float v100 = lut[(r1*N*N + g0*N + b0)*3 + c];
                float v101 = lut[(r1*N*N + g0*N + b1)*3 + c];
                float v110 = lut[(r1*N*N + g1*N + b0)*3 + c];
                float v111 = lut[(r1*N*N + g1*N + b1)*3 + c];

                result[c] = (1-dr)*(1-dg)*(1-db)*v000
                          + (1-dr)*(1-dg)*   db *v001
                          + (1-dr)*   dg *(1-db)*v010
                          + (1-dr)*   dg *   db *v011
                          +    dr *(1-dg)*(1-db)*v100
                          +    dr *(1-dg)*   db *v101
                          +    dr *   dg *(1-db)*v110
                          +    dr *   dg *   db *v111;
            }

            int alpha = p & 0xFF000000;
            out[i] = alpha
                | (clamp255(Math.round(result[0] * 255f)) << 16)
                | (clamp255(Math.round(result[1] * 255f)) <<  8)
                |  clamp255(Math.round(result[2] * 255f));
        }
        return out;
    }

    // ── Blend ─────────────────────────────────────────────────────────────────

    private static int[] blend(int[] a, int[] b, float s) {
        int[] out = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            int pa = a[i], pb = b[i];
            int ar = (pa >> 16) & 0xFF, br = (pb >> 16) & 0xFF;
            int ag = (pa >>  8) & 0xFF, bg = (pb >>  8) & 0xFF;
            int ab = ( pa      ) & 0xFF, bb = ( pb      ) & 0xFF;
            int alpha = pa & 0xFF000000;
            out[i] = alpha
                | (clamp255(Math.round(ar * (1-s) + br * s)) << 16)
                | (clamp255(Math.round(ag * (1-s) + bg * s)) <<  8)
                |  clamp255(Math.round(ab * (1-s) + bb * s));
        }
        return out;
    }

    private static int clamp255(int v) {
        return Math.max(0, Math.min(255, v));
    }
}
