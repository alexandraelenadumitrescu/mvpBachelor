package com.photomatch.api;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class BlurDetectResponse {
    public List<BlurRegion> regions;
    @SerializedName("image_width")  public int imageWidth;
    @SerializedName("image_height") public int imageHeight;
}
