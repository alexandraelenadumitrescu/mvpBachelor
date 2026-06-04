package com.photomatch.api;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class ScrapeResponse {
    @SerializedName("page_url")           public String                 pageUrl;
    @SerializedName("total_images_found") public int                    totalImagesFound;
    @SerializedName("images_processed")  public int                    imagesProcessed;
    @SerializedName("images_with_risk")  public int                    imagesWithRisk;
    @SerializedName("skipped_count")     public int                    skippedCount;
    public String                        detector;
    public List<ScrapedImageResult>      results;
}
