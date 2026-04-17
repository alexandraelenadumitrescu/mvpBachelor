package com.photomatch.ml;

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

public class DefectDetector {

    private static final String MODEL_FILE = "defect_head.tflite";

    private static final int INPUT_SIZE = 224;
    private static final int NUM_CHANNELS = 3;
    // float32 = 4 bytes; input tensor: [1, 224, 224, 3]
    private static final int INPUT_BUFFER_SIZE = 1 * INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS * 4;

    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD  = {0.229f, 0.224f, 0.225f};

    private static final int OUTPUT_SIZE = 5; // blur, noise, overexposure, underexposure, compression

    private final Interpreter interpreter;

    public DefectDetector(Context context) {
        try {
            MappedByteBuffer modelBuffer = loadModelFile(context);
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            interpreter = new Interpreter(modelBuffer, options);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load defect_head.tflite from assets", e);
        }
    }

    /**
     * Runs defect detection on the given bitmap.
     *
     * @param bitmap source image (any size; will be resized to 224x224 internally)
     * @return float[5] scores: [blur, noise, overexposure, underexposure, compression]
     */
    public float[] detect(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        ByteBuffer inputBuffer = bitmapToInputBuffer(resized);

        float[][] output = new float[1][OUTPUT_SIZE];
        interpreter.run(inputBuffer, output);

        return output[0];
    }

    public void close() {
        interpreter.close();
    }

    // --- private helpers ---

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Converts a 224x224 bitmap to a float32 ByteBuffer in NHWC order.
     * Pixels are normalized per-channel: (value/255 - mean) / std.
     */
    private ByteBuffer bitmapToInputBuffer(Bitmap bitmap) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(INPUT_BUFFER_SIZE);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >>  8) & 0xFF) / 255.0f;
            float b = ( pixel        & 0xFF) / 255.0f;

            buffer.putFloat((r - MEAN[0]) / STD[0]);
            buffer.putFloat((g - MEAN[1]) / STD[1]);
            buffer.putFloat((b - MEAN[2]) / STD[2]);
        }

        buffer.rewind();
        return buffer;
    }

    // --- inner class ---

    public static class DefectResult {
        public final float blur;
        public final float noise;
        public final float overexposure;
        public final float underexposure;
        public final float compression;

        public DefectResult(float[] scores) {
            this.blur         = scores[0];
            this.noise        = scores[1];
            this.overexposure = scores[2];
            this.underexposure = scores[3];
            this.compression  = scores[4];
        }

        @Override
        public String toString() {
            return String.format(
                "BLUR %.2f | NOISE %.2f | OVER %.2f | UNDER %.2f | COMP %.2f",
                blur, noise, overexposure, underexposure, compression
            );
        }
    }
}
