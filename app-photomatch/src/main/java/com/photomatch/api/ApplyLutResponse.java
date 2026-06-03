package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

public class ApplyLutResponse {
    @SerializedName("final_b64")  public String  finalB64;
    @SerializedName("lut_cached") public boolean lutCached;
    @SerializedName("note")       public String  note;
}
