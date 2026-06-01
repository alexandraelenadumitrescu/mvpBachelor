package com.photomatch.lite.api;

import java.util.List;

public class PhotoEmbeddings {
    public int photo_index;
    public List<List<Float>> embeddings;

    public PhotoEmbeddings(int photo_index, List<List<Float>> embeddings) {
        this.photo_index = photo_index;
        this.embeddings  = embeddings;
    }
}
