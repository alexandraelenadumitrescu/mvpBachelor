package com.photomatch.api;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class ScrapedImageResult {
    @SerializedName("source_url")      public String      sourceUrl;
    @SerializedName("thumbnail_b64")   public String      thumbnailB64;
    @SerializedName("original_width")  public int         originalWidth;
    @SerializedName("original_height") public int         originalHeight;
    @SerializedName("thumbnail_width")  public int        thumbnailWidth;
    @SerializedName("thumbnail_height") public int        thumbnailHeight;
    public List<BlurRegion>            regions;
    @SerializedName("risk_score")      public float       riskScore;
    @SerializedName("risk_label")      public String      riskLabel;
}
