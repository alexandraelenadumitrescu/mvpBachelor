package com.photomatch.lite.face;

import android.graphics.Bitmap;
import java.util.ArrayList;
import java.util.List;

public class FaceClusterer {

    public static final float THRESHOLD = 0.75f;

    public static class FaceEmbedding {
        public final String photoUri;
        public final Bitmap crop;
        public final float[] embedding;
        public final int faceIndex;

        public FaceEmbedding(String photoUri, Bitmap crop, float[] embedding, int faceIndex) {
            this.photoUri  = photoUri;
            this.crop      = crop;
            this.embedding = embedding;
            this.faceIndex = faceIndex;
        }
    }

    public static class FaceCluster {
        public int personIndex;
        public final List<FaceEmbedding> faces = new ArrayList<>();
    }

    public static List<FaceCluster> cluster(List<FaceEmbedding> faces) {
        return cluster(faces, THRESHOLD);
    }

    public static List<FaceCluster> cluster(List<FaceEmbedding> faces, float threshold) {
        int n = faces.size();
        boolean[] assigned = new boolean[n];
        List<FaceCluster> clusters = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            if (assigned[i]) continue;
            FaceCluster c = new FaceCluster();
            c.faces.add(faces.get(i));
            assigned[i] = true;
            for (int j = i + 1; j < n; j++) {
                if (!assigned[j] && dot(faces.get(i).embedding, faces.get(j).embedding) > threshold) {
                    c.faces.add(faces.get(j));
                    assigned[j] = true;
                }
            }
            clusters.add(c);
        }

        clusters.sort((a, b) -> Integer.compare(b.faces.size(), a.faces.size()));
        int idx = 1;
        for (FaceCluster c : clusters) c.personIndex = (c.faces.size() > 1) ? idx++ : 0;
        return clusters;
    }

    private static float dot(float[] a, float[] b) {
        float s = 0f;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }
}
