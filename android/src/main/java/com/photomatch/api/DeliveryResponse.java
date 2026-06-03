package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

public class DeliveryResponse {
    @SerializedName("clusters_found") public int clustersFound;
    public int matched;
    @SerializedName("emails_sent") public int sent;
    public int failed;
}
