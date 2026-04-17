package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

public class StyleUploadResponse {
    @SerializedName("session_id")     public String sessionId;
    @SerializedName("vectors_stored") public int    vectorsStored;
}
