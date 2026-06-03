package com.photomatch.base;

/**
 * Base for activities that make server calls.
 * Inherits executor lifecycle and setProcessing() from BaseLocalActivity.
 * Use api() from BaseActivity to reach the Retrofit service.
 */
public abstract class BaseServerActivity extends BaseLocalActivity {
    // No additional state needed — executor + setProcessing() from BaseLocalActivity,
    // api() from BaseActivity. Each subclass drives its own async flow.
}
