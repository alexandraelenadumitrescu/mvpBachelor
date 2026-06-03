package com.photomatch.api;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class ClusterRequest {
    @SerializedName("vectors")    public List<List<Float>> vectors;
    @SerializedName("n_clusters") public Integer           nClusters;  // null → auto-detect
}
