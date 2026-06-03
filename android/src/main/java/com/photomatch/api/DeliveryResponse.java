package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

public class DeliveryResponse {
    public int matched;
    @SerializedName("emails_sent") public int sent;
    public int failed;
}
