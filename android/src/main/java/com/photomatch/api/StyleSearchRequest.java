package com.photomatch.api;

import com.google.gson.annotations.SerializedName;
import java.util.ArrayList;
import java.util.List;

public class StyleSearchRequest {
    @SerializedName("vector")     public List<Float> vector;
    @SerializedName("session_id") public String      sessionId;

    public StyleSearchRequest(float[] vec, String sessionId) {
        this.vector = new ArrayList<>(vec.length);
        for (float v : vec) this.vector.add(v);
        this.sessionId = sessionId;
    }
}
