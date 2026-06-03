package com.photomatch.ml;

import android.clip.cpp.CLIPAndroid;
import android.content.Context;
import android.graphics.Bitmap;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.concurrent.Executor;

public class CLIPEncoder {

    public interface LoadCallback {
        void onLoaded();
        void onError(Exception e);
    }

    private static final int NUM_THREADS = 4;
    private static final int VERBOSITY = 1;
    private static final int EMBEDDING_DIM = 512;
    private static final int IMAGE_SIZE = 224;

    private final CLIPAndroid clipAndroid = new CLIPAndroid();
    private volatile boolean loaded = false;
    private final String resolvedModelPath;

    /**
     * Resolves the model file path. Does NOT load the model — call loadAsync() for that.
     *
     * @throws IllegalStateException if clip_model.gguf is not found in either search location.
     */
    public CLIPEncoder(Context context) {
        String internalPath = context.getFilesDir().getAbsolutePath() + "/clip_model.gguf";
        String tmpPath = "/data/local/tmp/clip_model.gguf";

        if (new File(internalPath).exists()) {
            resolvedModelPath = internalPath;
        } else if (new File(tmpPath).exists()) {
            resolvedModelPath = tmpPath;
        } else {
            throw new IllegalStateException(
                "clip_model.gguf not found. Checked:\n1. " + internalPath + "\n2. " + tmpPath
            );
        }
    }

    /**
     * Loads the CLIP model on the provided executor. Safe to call from the main thread.
     */
    public void loadAsync(Executor executor, LoadCallback callback) {
        executor.execute(() -> {
            try {
                clipAndroid.load(resolvedModelPath, VERBOSITY);
                loaded = true;
                callback.onLoaded();
            } catch (Exception e) {
                callback.onError(e);
            }
        });
    }

    /**
     * Encodes a bitmap into a 512-dim L2-normalized CLIP embedding.
     * Must be called after loadAsync() has completed successfully.
     * Call from a background thread — blocks during JNI inference.
     *
     * @param bitmap source image (any size; resized to 224×224 internally)
     * @return float[512] L2-normalized embedding vector
     */
    public float[] encode(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true);
        ByteBuffer buffer = bitmapToByteBuffer(resized);

        float[] embedding = clipAndroid.encodeImageNoResize(
            buffer, IMAGE_SIZE, IMAGE_SIZE, NUM_THREADS, EMBEDDING_DIM, true
        );

        return l2normalize(embedding);
    }

    /** Returns true if the model has been loaded successfully via loadAsync(). */
    public boolean isLoaded() {
        return loaded;
    }

    /** Releases JNI resources. Call when done with this encoder. */
    public void close() {
        clipAndroid.close();
    }

    // --- private helpers ---

    /**
     * Converts a bitmap to a direct ByteBuffer in raw RGB order (3 bytes/pixel, no alpha).
     * Matches the layout expected by CLIPAndroid.encodeImageNoResize().
     */
    private ByteBuffer bitmapToByteBuffer(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 3);

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int pixel : pixels) {
            buffer.put((byte) ((pixel >> 16) & 0xFF)); // R
            buffer.put((byte) ((pixel >>  8) & 0xFF)); // G
            buffer.put((byte) ( pixel        & 0xFF)); // B
        }

        buffer.rewind();
        return buffer;
    }

    private float[] l2normalize(float[] vec) {
        float norm = 0;
        for (float v : vec) norm += v * v;
        norm = (float) Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < vec.length; i++) vec[i] /= norm;
        }
        return vec;
    }
}
