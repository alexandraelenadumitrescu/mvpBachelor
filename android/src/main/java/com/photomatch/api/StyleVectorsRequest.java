package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

import java.util.ArrayList;
import java.util.List;

public class StyleVectorsRequest {
    @SerializedName("vectors")    public List<List<Float>> vectors;
    @SerializedName("session_id") public String            sessionId;  // null → new session

    public StyleVectorsRequest(List<float[]> vecs, String sessionId) {
        this.vectors   = new ArrayList<>(vecs.size());
        this.sessionId = sessionId;
        for (float[] v : vecs) {
            List<Float> row = new ArrayList<>(v.length);
            for (float f : v) row.add(f);
            this.vectors.add(row);
        }
    }
}
