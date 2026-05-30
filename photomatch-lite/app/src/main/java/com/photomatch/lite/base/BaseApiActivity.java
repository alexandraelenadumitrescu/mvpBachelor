package com.photomatch.lite.base;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.photomatch.lite.R;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public abstract class BaseApiActivity<TReq, TRes> extends AppCompatActivity {

    private ProgressBar progressBar;

    protected abstract View       onCreateLayout(LayoutInflater inflater);
    protected abstract void       onBindViews();
    protected abstract TReq       buildRequest();
    protected abstract Call<TRes> callApi(TReq request);
    protected abstract void       onSuccess(TRes response);

    @Override
    protected final void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(onCreateLayout(getLayoutInflater()));
        progressBar = findViewById(R.id.progressBar);
        onBindViews();
    }

    protected final void submit() {
        setLoading(true);
        callApi(buildRequest()).enqueue(new Callback<TRes>() {
            @Override
            public void onResponse(Call<TRes> call, Response<TRes> response) {
                setLoading(false);
                if (response.isSuccessful() && response.body() != null) onSuccess(response.body());
                else showError("Server error " + response.code());
            }

            @Override
            public void onFailure(Call<TRes> call, Throwable t) {
                setLoading(false);
                showError(t.getMessage());
            }
        });
    }

    private void setLoading(boolean on) {
        progressBar.setVisibility(on ? View.VISIBLE : View.GONE);
    }

    protected void showError(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show();
    }
}
