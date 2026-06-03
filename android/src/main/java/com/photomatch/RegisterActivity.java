package com.photomatch;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.photomatch.api.ApiClient;
import com.photomatch.api.RegisterRequest;
import com.photomatch.api.TokenResponse;
import com.photomatch.api.UserProfile;
import com.photomatch.base.BaseActivity;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class RegisterActivity extends BaseActivity {

    private EditText    etEmail;
    private EditText    etPassword;
    private EditText    etConfirm;
    private Button      btnRegister;
    private ProgressBar progressBar;
    private TextView    tvError;

    @Override protected boolean requiresAuth() { return false; }

    @Override
    protected void onStart() {
        super.onStart();
        if (ApiClient.isLoggedIn()) {
            startActivity(new Intent(this, MainActivity.class));
            finish();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        etEmail     = findViewById(R.id.etEmail);
        etPassword  = findViewById(R.id.etPassword);
        etConfirm   = findViewById(R.id.etConfirmPassword);
        btnRegister = findViewById(R.id.btnRegister);
        progressBar = findViewById(R.id.progressBar);
        tvError     = findViewById(R.id.tvError);

        btnRegister.setOnClickListener(v -> attemptRegister());

        findViewById(R.id.tvLogin).setOnClickListener(v -> {
            startActivity(new Intent(this, LoginActivity.class));
            finish();
        });
    }

    private void attemptRegister() {
        String email    = etEmail.getText().toString().trim();
        String password = etPassword.getText().toString();
        String confirm  = etConfirm.getText().toString();

        if (email.isEmpty() || password.isEmpty()) {
            showFormError("Email si parola sunt obligatorii");
            return;
        }
        if (!password.equals(confirm)) {
            showFormError("Parolele nu coincid");
            return;
        }
        if (password.length() < 6) {
            showFormError("Parola trebuie sa aiba cel putin 6 caractere");
            return;
        }

        tvError.setVisibility(View.GONE);
        btnRegister.setEnabled(false);
        progressBar.setVisibility(View.VISIBLE);

        api().register(new RegisterRequest(email, password))
            .enqueue(new Callback<UserProfile>() {
                @Override
                public void onResponse(Call<UserProfile> call, Response<UserProfile> response) {
                    if (response.isSuccessful()) {
                        // Auto-login after successful registration
                        autoLogin(email, password);
                    } else {
                        progressBar.setVisibility(View.GONE);
                        btnRegister.setEnabled(true);
                        String msg = response.code() == 400
                            ? "Email deja inregistrat"
                            : "Eroare: HTTP " + response.code();
                        showFormError(msg);
                    }
                }

                @Override
                public void onFailure(Call<UserProfile> call, Throwable t) {
                    progressBar.setVisibility(View.GONE);
                    btnRegister.setEnabled(true);
                    showFormError("Eroare de retea: " + t.getMessage());
                }
            });
    }

    private void autoLogin(String email, String password) {
        api().login(email, password).enqueue(new Callback<TokenResponse>() {
            @Override
            public void onResponse(Call<TokenResponse> call, Response<TokenResponse> response) {
                progressBar.setVisibility(View.GONE);
                if (response.isSuccessful() && response.body() != null
                        && response.body().accessToken != null) {
                    ApiClient.saveToken(response.body().accessToken);
                    Intent intent = new Intent(RegisterActivity.this, MainActivity.class);
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_NEW_TASK);
                    startActivity(intent);
                    finish();
                } else {
                    // Registration OK but login failed — redirect to login screen
                    startActivity(new Intent(RegisterActivity.this, LoginActivity.class));
                    finish();
                }
            }

            @Override
            public void onFailure(Call<TokenResponse> call, Throwable t) {
                progressBar.setVisibility(View.GONE);
                startActivity(new Intent(RegisterActivity.this, LoginActivity.class));
                finish();
            }
        });
    }

    private void showFormError(String msg) {
        tvError.setText(msg);
        tvError.setVisibility(View.VISIBLE);
    }
}
