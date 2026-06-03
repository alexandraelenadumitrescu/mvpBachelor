package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

public class StyleSearchResponse {
    @SerializedName("retrieved")       public String  retrieved;
    @SerializedName("similarity")      public float   similarity;
    @SerializedName("style_matched")   public boolean styleMatched;
    @SerializedName("style_fallback")  public boolean styleFallback;
    @SerializedName("lut_cached")      public boolean lutCached;
}
