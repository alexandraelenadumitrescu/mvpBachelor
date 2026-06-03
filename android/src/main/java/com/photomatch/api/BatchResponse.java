package com.photomatch.api;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class BatchResponse {
    @SerializedName("results")   public List<BatchResult> results;
    @SerializedName("processed") public int processed;
    @SerializedName("failed")    public int failed;
}
