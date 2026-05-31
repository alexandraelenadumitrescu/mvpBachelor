package com.photomatch.lite.api;

import java.util.List;

public class DeliveryResponse {
    public int                 matched;
    public int                 emails_sent;
    public int                 failed;
    public List<DeliveryDetail> details;
}
