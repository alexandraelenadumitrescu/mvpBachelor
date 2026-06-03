package com.photomatch;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.photomatch.api.ApiClient;
import com.photomatch.api.TokenResponse;
import com.photomatch.base.BaseActivity;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class LoginActivity extends BaseActivity {

    private EditText    etEmail;
    private EditText    etPassword;
    private Button      btnLogin;
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
        setContentView(R.layout.activity_login);

        etEmail     = findViewById(R.id.etEmail);
        etPassword  = findViewById(R.id.etPassword);
        btnLogin    = findViewById(R.id.btnLogin);
        progressBar = findViewById(R.id.progressBar);

        findViewById(R.id.tvRegister).setOnClickListener(v -> {
            startActivity(new Intent(this, RegisterActivity.class));
            finish();
        });
        tvError     = findViewById(R.id.tvError);

        etEmail.setText("a@a.com");
        etPassword.setText("string");

        btnLogin.setOnClickListener(v -> attemptLogin());
    }

    private void attemptLogin() {
        String email    = etEmail.getText().toString().trim();
        String password = etPassword.getText().toString();

        if (email.isEmpty() || password.isEmpty()) {
            tvError.setText("Email si parola sunt obligatorii");
            tvError.setVisibility(View.VISIBLE);
            return;
        }

        tvError.setVisibility(View.GONE);
        btnLogin.setEnabled(false);
        progressBar.setVisibility(View.VISIBLE);

        ApiClient.service().login(email, password).enqueue(new Callback<TokenResponse>() {
            @Override
            public void onResponse(Call<TokenResponse> call, Response<TokenResponse> response) {
                progressBar.setVisibility(View.GONE);
                btnLogin.setEnabled(true);

                if (response.isSuccessful() && response.body() != null
                        && response.body().accessToken != null) {
                    ApiClient.saveToken(response.body().accessToken);
                    Intent intent = new Intent(LoginActivity.this, MainActivity.class);
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_NEW_TASK);
                    startActivity(intent);
                    finish();
                } else {
                    tvError.setText("Email sau parola incorecta");
                    tvError.setVisibility(View.VISIBLE);
                }
            }

            @Override
            public void onFailure(Call<TokenResponse> call, Throwable t) {
                progressBar.setVisibility(View.GONE);
                btnLogin.setEnabled(true);
                tvError.setText("Eroare de retea: " + t.getMessage());
                tvError.setVisibility(View.VISIBLE);
            }
        });
    }
}
