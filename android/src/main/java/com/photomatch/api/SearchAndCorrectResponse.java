package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

public class SearchAndCorrectResponse {
    @SerializedName("retrieved")             public String  retrieved;
    @SerializedName("similarity")            public float   similarity;
    @SerializedName("lut_cached")            public boolean lutCached;
    @SerializedName("match_aesthetic_score") public float   matchAestheticScore;
}
