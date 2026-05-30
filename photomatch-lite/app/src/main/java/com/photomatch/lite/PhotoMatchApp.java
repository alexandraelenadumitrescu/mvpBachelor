package com.photomatch.lite;

import android.app.Application;
import com.photomatch.lite.api.ApiClient;

public class PhotoMatchApp extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        ApiClient.init(this);
    }
}
