package com.photomatch;

import android.app.Application;
import com.photomatch.api.ApiClient;

public class PhotoMatchApp extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        ApiClient.init(this);
    }
}
