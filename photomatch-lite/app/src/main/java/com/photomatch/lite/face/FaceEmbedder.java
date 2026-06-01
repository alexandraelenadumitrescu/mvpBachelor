package com.photomatch.lite.face;

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

public class FaceEmbedder {

    private static final int INPUT_SIZE        = 160;
    private static final int OUTPUT_DIM        = 128;
    private static final int INPUT_BUFFER_SIZE = INPUT_SIZE * INPUT_SIZE * 3 * 4;

    private final Interpreter interpreter;

    public FaceEmbedder(Context context) {
        try {
            AssetFileDescriptor fd = context.getAssets().openFd("facenet.tflite");
            FileChannel channel = new FileInputStream(fd.getFileDescriptor()).getChannel();
            MappedByteBuffer buf = channel.map(FileChannel.MapMode.READ_ONLY,
                fd.getStartOffset(), fd.getDeclaredLength());
            Interpreter.Options opts = new Interpreter.Options();
            opts.setNumThreads(4);
            interpreter = new Interpreter(buf, opts);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load facenet.tflite", e);
        }
    }

    public float[] embed(Bitmap faceCrop) {
        Bitmap resized = Bitmap.createScaledBitmap(faceCrop, INPUT_SIZE, INPUT_SIZE, true);
        ByteBuffer buf = ByteBuffer.allocateDirect(INPUT_BUFFER_SIZE);
        buf.order(ByteOrder.nativeOrder());
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        for (int p : pixels) {
            buf.putFloat((((p >> 16) & 0xFF) - 127.5f) / 128.0f);
            buf.putFloat((((p >>  8) & 0xFF) - 127.5f) / 128.0f);
            buf.putFloat((( p        & 0xFF) - 127.5f) / 128.0f);
        }
        buf.rewind();
        if (resized != faceCrop) resized.recycle();
        float[][] out = new float[1][OUTPUT_DIM];
        interpreter.run(buf, out);
        return l2normalize(out[0]);
    }

    public void close() { interpreter.close(); }

    private float[] l2normalize(float[] v) {
        float norm = 0f;
        for (float x : v) norm += x * x;
        norm = (float) Math.sqrt(norm);
        if (norm > 0f) for (int i = 0; i < v.length; i++) v[i] /= norm;
        return v;
    }
}
