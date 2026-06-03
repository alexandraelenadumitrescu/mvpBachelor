package com.photomatch.api;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class BatchRequest {
    @SerializedName("vectors")    public List<List<Float>> vectors;
    @SerializedName("session_id") public String sessionId;  // nullable
}
