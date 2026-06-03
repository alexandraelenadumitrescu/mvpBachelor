package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

import java.util.Map;

public class ProcessResponse {
    @SerializedName("original_b64")  public String originalB64;
    @SerializedName("corrected_b64") public String correctedB64;
    @SerializedName("final_b64")     public String finalB64;
    @SerializedName("defects")       public Map<String, Float> defects;
    @SerializedName("retrieved")     public String retrieved;
    @SerializedName("similarity")    public float similarity;
    @SerializedName("raw_b64")       public String rawB64;
    @SerializedName("edited_b64")    public String editedB64;
    @SerializedName("note")                  public String note;  // "" when LUT applied; message when CLAHE only
    @SerializedName("match_aesthetic_score") public float  matchAestheticScore;
}
