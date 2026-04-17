package com.photomatch.api;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;

import okhttp3.OkHttpClient;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

import java.io.ByteArrayOutputStream;
import java.util.concurrent.TimeUnit;

public class ApiClient {

    public static final String PREFS_NAME    = "photomatch_prefs";
    public static final String KEY_SERVER_IP = "server_ip";
    public static final String DEFAULT_IP    = "192.168.1.132";

    private static Context   appContext;
    private static ApiClient instance;
    private        ApiService apiService;

    public static void init(Context context) {
        appContext = context.getApplicationContext();
    }

    public static String getServerIp() {
        SharedPreferences prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        return prefs.getString(KEY_SERVER_IP, DEFAULT_IP);
    }

    public static void saveServerIp(String ip) {
        appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit().putString(KEY_SERVER_IP, ip).apply();
        instance = null;
    }

    private ApiClient() {
        OkHttpClient okHttpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(600, TimeUnit.SECONDS)   // batch: up to 100 images × ~2s each
            .writeTimeout(30, TimeUnit.SECONDS)
            .build();

        String baseUrl = "http://" + getServerIp() + ":8000/";
        Retrofit retrofit = new Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build();
        apiService = retrofit.create(ApiService.class);
    }

    public static synchronized ApiClient getInstance() {
        if (instance == null) {
            instance = new ApiClient();
        }
        return instance;
    }

    public ApiService getService() {
        return apiService;
    }

    // --- static bitmap helpers ---

    /** Decodes a Base64 string into a Bitmap. */
    public static Bitmap base64ToBitmap(String base64) {
        byte[] bytes = Base64.decode(base64, Base64.DEFAULT);
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
    }

    /** Encodes a Bitmap to a Base64 string (JPEG, quality 90). */
    public static String bitmapToBase64(Bitmap bitmap) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out);
        return Base64.encodeToString(out.toByteArray(), Base64.DEFAULT);
    }
}
