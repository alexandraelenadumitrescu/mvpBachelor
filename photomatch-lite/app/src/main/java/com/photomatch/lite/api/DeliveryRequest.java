package com.photomatch.lite.api;

import android.net.Uri;
import java.util.List;

public class DeliveryRequest {
    public String    employeesUrl;
    public List<Uri> photos;

    public DeliveryRequest(String employeesUrl, List<Uri> photos) {
        this.employeesUrl = employeesUrl;
        this.photos       = photos;
    }
}
