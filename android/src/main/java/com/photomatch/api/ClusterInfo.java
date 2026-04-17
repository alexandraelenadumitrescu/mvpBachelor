package com.photomatch.api;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class ClusterInfo {
    @SerializedName("cluster_id") public int          clusterID;
    @SerializedName("indices")    public List<Integer> indices;
}
