package com.photomatch.api;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;

import java.io.ByteArrayOutputStream;
import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class ApiClient {

    public static final String PREFS_NAME    = "photomatch_prefs";
    public static final String KEY_SERVER_IP = "server_ip";
    public static final String KEY_TOKEN     = "auth_token";
    public static final String DEFAULT_IP    = "192.168.1.132";

    private static Context   appContext;
    private static ApiClient instance;
    private        ApiService apiService;

    public static void init(Context context) {
        appContext = context.getApplicationContext();
    }

    // --- server IP ---

    public static String getServerIp() {
        return prefs().getString(KEY_SERVER_IP, DEFAULT_IP);
    }

    public static void saveServerIp(String ip) {
        prefs().edit().putString(KEY_SERVER_IP, ip).apply();
        instance = null; // force singleton recreation with new base URL
    }

    // --- auth token ---

    public static void saveToken(String token) {
        prefs().edit().putString(KEY_TOKEN, token).apply();
    }

    public static String getToken() {
        return prefs().getString(KEY_TOKEN, null);
    }

    public static void clearToken() {
        prefs().edit().remove(KEY_TOKEN).apply();
    }

    public static boolean isLoggedIn() {
        return getToken() != null;
    }

    // --- singleton access ---

    public static ApiService service() {
        return getInstance().getService();
    }

    public static synchronized ApiClient getInstance() {
        if (instance == null) instance = new ApiClient();
        return instance;
    }

    public ApiService getService() {
        return apiService;
    }

    // --- constructor ---

    private ApiClient() {
        OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(600, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .addInterceptor(chain -> {
                Request req = chain.request();
                String path = req.url().encodedPath();
                String token = getToken();
                // skip auth header for login and register endpoints
                if (token == null || path.contains("/auth/login") || path.contains("/auth/register"))
                    return chain.proceed(req);
                return chain.proceed(req.newBuilder()
                    .header("Authorization", "Bearer " + token).build());
            })
            .build();

        Retrofit retrofit = new Retrofit.Builder()
            .baseUrl("http://" + getServerIp() + ":8000/")
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build();
        apiService = retrofit.create(ApiService.class);
    }

    // --- bitmap helpers ---

    public static Bitmap base64ToBitmap(String base64) {
        byte[] bytes = Base64.decode(base64, Base64.DEFAULT);
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
    }

    public static String bitmapToBase64(Bitmap bitmap) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out);
        return Base64.encodeToString(out.toByteArray(), Base64.DEFAULT);
    }

    // --- private ---

    private static SharedPreferences prefs() {
        return appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }
}
