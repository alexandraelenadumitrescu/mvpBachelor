package com.photomatch;

import android.graphics.Bitmap;

import java.util.ArrayList;
import java.util.List;

/**
 * Greedy cosine-similarity clustering for face embeddings.
 * Embeddings must be L2-normalised — cosine similarity equals dot product.
 */
public class FaceClusterer {

    static final float FACE_SIMILARITY_THRESHOLD = 0.75f;

    // --- Data classes ---

    public static class FaceEmbedding {
        public final String photoUri;
        public final Bitmap crop;       // 56×56 face crop, caller responsible for lifecycle
        public final float[] embedding; // 128-dim, L2-normalised
        public final int faceIndex;     // index within the source photo

        public FaceEmbedding(String photoUri, Bitmap crop, float[] embedding, int faceIndex) {
            this.photoUri  = photoUri;
            this.crop      = crop;
            this.embedding = embedding;
            this.faceIndex = faceIndex;
        }
    }

    public static class FaceCluster {
        public int personIndex;                 // 1-based; 0 = "Unmatched"
        public final List<FaceEmbedding> faces = new ArrayList<>();
    }

    // --- Clustering ---

    /**
     * Groups faces into person clusters using greedy nearest-neighbour merging.
     *
     * @param faces List of L2-normalised embeddings to cluster
     * @return Clusters sorted by size (largest first); singletons last with personIndex=0
     */
    public static List<FaceCluster> cluster(List<FaceEmbedding> faces) {
        return cluster(faces, FACE_SIMILARITY_THRESHOLD);
    }

    public static List<FaceCluster> cluster(List<FaceEmbedding> faces, float threshold) {
        int n = faces.size();
        boolean[] assigned = new boolean[n];
        List<FaceCluster> clusters = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            if (assigned[i]) continue;
            FaceCluster cluster = new FaceCluster();
            cluster.faces.add(faces.get(i));
            assigned[i] = true;
            for (int j = i + 1; j < n; j++) {
                if (!assigned[j]) {
                    float sim = dotProduct(faces.get(i).embedding, faces.get(j).embedding);
                    if (sim > threshold) {
                        cluster.faces.add(faces.get(j));
                        assigned[j] = true;
                    }
                }
            }
            clusters.add(cluster);
        }

        // Sort: larger clusters first
        clusters.sort((a, b) -> Integer.compare(b.faces.size(), a.faces.size()));

        // Assign person indices: 1-based for multi-face clusters, 0 for singletons
        int personIdx = 1;
        for (FaceCluster c : clusters) {
            c.personIndex = (c.faces.size() > 1) ? personIdx++ : 0;
        }

        return clusters;
    }

    private static float dotProduct(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }
}
