package com.photomatch.api;

import com.google.gson.annotations.SerializedName;
import java.util.ArrayList;
import java.util.List;

public class SearchAndCorrectRequest {
    @SerializedName("vector") public List<Float> vector;
    @SerializedName("top_k")  public int         topK = 5;

    public SearchAndCorrectRequest(float[] vec) {
        vector = new ArrayList<>(vec.length);
        for (float v : vec) vector.add(v);
    }
}
