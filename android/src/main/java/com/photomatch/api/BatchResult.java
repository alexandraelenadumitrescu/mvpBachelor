package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

public class BatchResult {
    @SerializedName("index")         public int     index;
    @SerializedName("retrieved")     public String  retrieved;
    @SerializedName("similarity")    public float   similarity;
    @SerializedName("corrected_b64")         public String  correctedB64;   // v1 server path
    @SerializedName("corrected_path")        public String  correctedPath;  // v2 local file path
    @SerializedName("match_aesthetic_score") public float   matchAestheticScore;
}
