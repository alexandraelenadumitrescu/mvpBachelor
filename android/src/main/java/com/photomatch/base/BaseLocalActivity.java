package com.photomatch.base;

import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public abstract class BaseLocalActivity extends BaseActivity {

    protected ExecutorService executor;

    // Subclasses can set these to get automatic show/hide behaviour in setProcessing()
    protected Button      primaryButton;
    protected ProgressBar progressBar;

    @Override
    protected void onCreate(android.os.Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        executor = Executors.newSingleThreadExecutor();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executor != null) executor.shutdownNow();
    }

    protected void setProcessing(boolean processing) {
        if (primaryButton != null) primaryButton.setEnabled(!processing);
        if (progressBar  != null) progressBar.setVisibility(processing ? View.VISIBLE : View.GONE);
    }
}
