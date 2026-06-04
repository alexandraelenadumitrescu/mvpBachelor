package com.photomatch.api;

public class ScrapeRequest {
    public String url;
    public String detector;

    public ScrapeRequest(String url, String detector) {
        this.url      = url;
        this.detector = detector;
    }
}
