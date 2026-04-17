package com.photomatch.ml;

import android.content.Context;
import android.util.Base64;

import com.photomatch.api.ApiClient;
import com.photomatch.api.LutResponse;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.ConcurrentHashMap;

import retrofit2.Response;

/**
 * Downloads 3D LUTs from the server and caches them on-device in
 * {@code getFilesDir()/luts/<basename>.lut}.
 *
 * A LUT is a flat float32 array of length {@code lutSize^3 * 3}
 * ordered as [r][g][b][channel] (row-major, matching NumPy's C order).
 */
public class LutCache {

    private static final String LUT_DIR = "luts";

    // In-memory cache: basename → flat float32 array
    private static final ConcurrentHashMap<String, float[]> memCache = new ConcurrentHashMap<>();

    /**
     * Returns the LUT for {@code basename}, downloading it if not already cached.
     * Returns {@code null} if the download fails (caller should fall back to CLAHE-only).
     * Must be called from a background thread.
     */
    public static float[] get(Context context, String basename) {
        // 1. In-memory hit
        float[] lut = memCache.get(basename);
        if (lut != null) return lut;

        // 2. Disk hit
        File lutFile = lutFile(context, basename);
        if (lutFile.exists()) {
            lut = readFromDisk(lutFile);
            if (lut != null) {
                memCache.put(basename, lut);
                return lut;
            }
        }

        // 3. Download from server
        try {
            Response<LutResponse> resp = ApiClient.getInstance()
                .getService().getLut(basename).execute();
            if (!resp.isSuccessful() || resp.body() == null) return null;

            byte[] bytes = Base64.decode(resp.body().lutB64, Base64.DEFAULT);
            lut = bytesToFloats(bytes);
            writeToDisk(lutFile, bytes);
            memCache.put(basename, lut);
            return lut;
        } catch (IOException e) {
            return null;
        }
    }

    // ── private helpers ──────────────────────────────────────────────────────

    private static File lutFile(Context context, String basename) {
        File dir = new File(context.getFilesDir(), LUT_DIR);
        //noinspection ResultOfMethodCallIgnored
        dir.mkdirs();
        return new File(dir, basename + ".lut");
    }

    private static float[] bytesToFloats(byte[] bytes) {
        FloatBuffer fb = ByteBuffer.wrap(bytes)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer();
        float[] arr = new float[fb.capacity()];
        fb.get(arr);
        return arr;
    }

    private static void writeToDisk(File file, byte[] rawBytes) {
        try (FileOutputStream fos = new FileOutputStream(file);
             DataOutputStream dos = new DataOutputStream(fos)) {
            dos.write(rawBytes);
        } catch (IOException ignored) {}
    }

    private static float[] readFromDisk(File file) {
        try {
            byte[] bytes = new byte[(int) file.length()];
            try (FileInputStream fis = new FileInputStream(file);
                 DataInputStream dis = new DataInputStream(fis)) {
                dis.readFully(bytes);
            }
            return bytesToFloats(bytes);
        } catch (IOException e) {
            return null;
        }
    }
}
