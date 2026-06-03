package com.photomatch;

import java.util.ArrayList;
import java.util.List;

/**
 * Greedy single-pass cosine-similarity clustering.
 * Input vectors must already be L2-normalised (CLIPEncoder guarantees this),
 * so cosine similarity = dot product.
 */
public class BurstClusterer {

    /** Photos with cosine similarity above this threshold are grouped together. */
    public static final float THRESHOLD = 0.92f;

    /**
     * Groups photo indices into clusters of near-identical shots.
     *
     * @param vectors List of 512-dim L2-normalised CLIP vectors (one per photo).
     * @return List of clusters; each cluster is a list of photo indices.
     *         Singletons (no near-duplicate found) are included as 1-element lists.
     */
    public static List<List<Integer>> cluster(List<float[]> vectors) {
        return cluster(vectors, THRESHOLD);
    }

    public static List<List<Integer>> cluster(List<float[]> vectors, float threshold) {
        int n = vectors.size();
        boolean[] assigned = new boolean[n];
        List<List<Integer>> clusters = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            if (assigned[i]) continue;
            List<Integer> group = new ArrayList<>();
            group.add(i);
            assigned[i] = true;
            float[] vi = vectors.get(i);
            for (int j = i + 1; j < n; j++) {
                if (!assigned[j] && dot(vi, vectors.get(j)) > threshold) {
                    group.add(j);
                    assigned[j] = true;
                }
            }
            clusters.add(group);
        }
        return clusters;
    }

    /** Dot product — equals cosine similarity for L2-normalised vectors. */
    private static float dot(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }
}
