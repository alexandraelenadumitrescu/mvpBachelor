package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

public class ApplyLutRequest {
    @SerializedName("image_b64")          public String imageB64;
    @SerializedName("retrieved_basename") public String retrievedBasename;
}
