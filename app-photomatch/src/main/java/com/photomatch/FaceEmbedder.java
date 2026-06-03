package com.photomatch;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Computes 128-dim L2-normalised face embeddings using FaceNet TFLite.
 *
 * Model: assets/facenet.tflite
 * Input:  [1, 160, 160, 3] float32, pixels normalised via (pixel - 127.5) / 128
 * Output: [1, 128] float32 embedding
 */
public class FaceEmbedder {

    private static final String MODEL_FILE   = "facenet.tflite";
    private static final int    INPUT_SIZE   = 160;
    private static final int    OUTPUT_DIM   = 128;
    private static final int    INPUT_BUFFER_SIZE = 1 * INPUT_SIZE * INPUT_SIZE * 3 * 4;

    private final Interpreter interpreter;

    public FaceEmbedder(Context context) {
        try {
            MappedByteBuffer modelBuffer = loadModelFile(context);
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            interpreter = new Interpreter(modelBuffer, options);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load facenet.tflite from assets", e);
        }
    }

    /**
     * Embeds a face crop bitmap into a 128-dim L2-normalised float vector.
     * Does not recycle the input bitmap.
     */
    public float[] embed(Bitmap faceCrop) {
        Bitmap resized = Bitmap.createScaledBitmap(faceCrop, INPUT_SIZE, INPUT_SIZE, true);
        ByteBuffer inputBuffer = bitmapToInputBuffer(resized);
        if (resized != faceCrop) resized.recycle();

        float[][] output = new float[1][OUTPUT_DIM];
        interpreter.run(inputBuffer, output);

        return l2normalize(output[0]);
    }

    public void close() {
        interpreter.close();
    }

    // --- private helpers (same pattern as DefectDetector) ---

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor fd = context.getAssets().openFd(MODEL_FILE);
        FileInputStream is = new FileInputStream(fd.getFileDescriptor());
        FileChannel channel = is.getChannel();
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
    }

    private ByteBuffer bitmapToInputBuffer(Bitmap bitmap) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(INPUT_BUFFER_SIZE);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            float r = (((pixel >> 16) & 0xFF) - 127.5f) / 128.0f;
            float g = (((pixel >>  8) & 0xFF) - 127.5f) / 128.0f;
            float b = (( pixel        & 0xFF) - 127.5f) / 128.0f;
            buffer.putFloat(r);
            buffer.putFloat(g);
            buffer.putFloat(b);
        }

        buffer.rewind();
        return buffer;
    }

    private float[] l2normalize(float[] vec) {
        float norm = 0f;
        for (float v : vec) norm += v * v;
        norm = (float) Math.sqrt(norm);
        if (norm > 0f) {
            for (int i = 0; i < vec.length; i++) vec[i] /= norm;
        }
        return vec;
    }
}
