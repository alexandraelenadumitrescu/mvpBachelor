package com.photomatch.ml;

public class HybridVectorBuilder {

    private static final float CLIP_WEIGHT   = 1.0f;
    private static final float DEFECT_WEIGHT = 5.0f;

    /**
     * Builds a 517-dim hybrid unit vector from a 512-dim CLIP embedding and
     * a 5-dim defect score vector.
     *
     * Steps:
     *  1. L2-normalize clipVector, scale by CLIP_WEIGHT (1.0)
     *  2. L2-normalize defectVector, scale by DEFECT_WEIGHT (5.0)
     *  3. Concatenate → float[517]
     *  4. L2-normalize the concatenated result
     *
     * @param clipVector   float[512] CLIP image embedding
     * @param defectVector float[5]  defect scores [blur, noise, over, under, compression]
     * @return float[517] unit-length hybrid vector
     */
    public static float[] build(float[] clipVector, float[] defectVector) {
        float[] scaledClip   = normalizeAndScale(clipVector.clone(),   CLIP_WEIGHT);
        float[] scaledDefect = normalizeAndScale(defectVector.clone(), DEFECT_WEIGHT);

        float[] hybrid = new float[scaledClip.length + scaledDefect.length]; // 517
        System.arraycopy(scaledClip,   0, hybrid, 0,                 scaledClip.length);
        System.arraycopy(scaledDefect, 0, hybrid, scaledClip.length, scaledDefect.length);

        return l2normalize(hybrid);
    }

    private static float[] normalizeAndScale(float[] vec, float weight) {
        l2normalize(vec);
        for (int i = 0; i < vec.length; i++) vec[i] *= weight;
        return vec;
    }

    private static float[] l2normalize(float[] vec) {
        float norm = 0;
        for (float v : vec) norm += v * v;
        norm = (float) Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < vec.length; i++) vec[i] /= norm;
        }
        return vec;
    }
}
