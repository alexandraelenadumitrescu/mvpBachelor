package com.photomatch.lite.ui;

import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import com.photomatch.lite.api.ApiClient;
import com.photomatch.lite.api.TokenResponse;
import com.photomatch.lite.api.UserCreate;
import com.photomatch.lite.base.BaseApiActivity;
import com.photomatch.lite.databinding.ActivityLoginBinding;
import retrofit2.Call;

public class LoginActivity extends BaseApiActivity<UserCreate, TokenResponse> {

    private ActivityLoginBinding binding;

    @Override
    protected View onCreateLayout(LayoutInflater inflater) {
        binding = ActivityLoginBinding.inflate(inflater);
        return binding.getRoot();
    }

    @Override
    protected void onBindViews() {
        binding.btnLogin.setOnClickListener(v -> submit());
    }

    @Override
    protected UserCreate buildRequest() {
        return new UserCreate(
            binding.etEmail.getText().toString().trim(),
            binding.etPassword.getText().toString()
        );
    }

    @Override
    protected Call<TokenResponse> callApi(UserCreate request) {
        return ApiClient.service().login(request.email, request.password);
    }

    @Override
    protected void onSuccess(TokenResponse response) {
        ApiClient.saveToken(response.access_token);
        startActivity(new Intent(this, MainActivity.class));
        finish();
    }
}
