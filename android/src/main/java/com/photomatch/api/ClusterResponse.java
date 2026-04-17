package com.photomatch.api;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class ClusterResponse {
    @SerializedName("clusters")         public List<ClusterInfo> clusters;
    @SerializedName("n_clusters")       public int               nClusters;
    @SerializedName("silhouette_score") public float             silhouetteScore;
}
