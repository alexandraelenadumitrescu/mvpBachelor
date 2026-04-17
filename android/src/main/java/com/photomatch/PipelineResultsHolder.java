package com.photomatch;

import android.net.Uri;

import java.util.ArrayList;
import java.util.List;

/**
 * Singleton holding the final photo set after a pipeline run.
 * Cleared when the user starts a new pipeline.
 */
public class PipelineResultsHolder {

    public static final PipelineResultsHolder instance = new PipelineResultsHolder();

    private PipelineResultsHolder() {}

    public static class PipelinePhoto {
        public Uri    originalUri;
        public String correctedBase64;  // null if no correction was applied
    }

    public final List<PipelinePhoto> photos = new ArrayList<>();

    public void clear() {
        photos.clear();
    }
}
