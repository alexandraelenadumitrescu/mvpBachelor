package com.photomatch.util;

import android.content.ContentResolver;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;

import java.io.IOException;
import java.io.InputStream;

public final class ImageUtils {

    private ImageUtils() {}

    /** Returns the smallest power-of-2 sample size that keeps the long side <= maxSide. */
    public static int computeSampleSize(int width, int height, int maxSide) {
        int inSampleSize = 1;
        int maxDim = Math.max(width, height);
        while (maxDim / (inSampleSize * 2) > maxSide) {
            inSampleSize *= 2;
        }
        return inSampleSize;
    }

    /** Decodes a content URI into a Bitmap scaled so the long side <= maxSide. */
    public static Bitmap decodeBitmap(ContentResolver cr, Uri uri, int maxSide) throws IOException {
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inJustDecodeBounds = true;
        try (InputStream is = cr.openInputStream(uri)) {
            BitmapFactory.decodeStream(is, null, opts);
        }
        opts.inSampleSize = computeSampleSize(opts.outWidth, opts.outHeight, maxSide);
        opts.inJustDecodeBounds = false;
        try (InputStream is = cr.openInputStream(uri)) {
            Bitmap bmp = BitmapFactory.decodeStream(is, null, opts);
            if (bmp == null) throw new IOException("Could not decode image: " + uri);
            return bmp;
        }
    }

    /** Decodes a file path into a Bitmap scaled so the long side <= maxSide. */
    public static Bitmap decodeBitmap(String path, int maxSide) throws IOException {
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(path, opts);
        opts.inSampleSize = computeSampleSize(opts.outWidth, opts.outHeight, maxSide);
        opts.inJustDecodeBounds = false;
        Bitmap bmp = BitmapFactory.decodeFile(path, opts);
        if (bmp == null) throw new IOException("Could not decode image: " + path);
        return bmp;
    }
}
