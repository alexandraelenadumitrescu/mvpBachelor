package com.photomatch.lite.api;

import java.util.List;

public class MatchRequest {
    public String employees_url;
    public List<PhotoEmbeddings> photos;

    public MatchRequest(String employees_url, List<PhotoEmbeddings> photos) {
        this.employees_url = employees_url;
        this.photos        = photos;
    }
}
