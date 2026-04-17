package com.photomatch;

import com.photomatch.api.ProcessResponse;

/** Static holder for the latest ProcessResponse — avoids passing large base64 strings via Intent. */
public class ResponseCache {
    public static ProcessResponse current;
}
