package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

public class LutResponse {
    /** Flat float32 byte array (lut_size^3 * 3 values) encoded as base64. */
    @SerializedName("lut_b64")  public String lutB64;
    /** Grid side length — always 17. */
    @SerializedName("lut_size") public int    lutSize;
}
