package com.photomatch.base;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.text.InputType;
import android.widget.EditText;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.photomatch.LoginActivity;
import com.photomatch.api.ApiClient;
import com.photomatch.api.ApiService;

public abstract class BaseActivity extends AppCompatActivity {

    /** Override to return false in pre-auth screens (Login, Register). */
    protected boolean requiresAuth() { return true; }

    @Override
    protected void onStart() {
        super.onStart();
        if (requiresAuth() && !ApiClient.isLoggedIn()) {
            startActivity(new Intent(this, LoginActivity.class));
            finish();
        }
    }

    protected ApiService api() {
        return ApiClient.service();
    }

    protected void showError(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show();
    }

    protected void showServerIpDialog() {
        EditText input = new EditText(this);
        input.setHint("ex: 10.33.128.137");
        input.setText(ApiClient.getServerIp());
        input.setInputType(InputType.TYPE_CLASS_TEXT);

        new AlertDialog.Builder(this)
            .setTitle("Server IP")
            .setMessage("Introdu IP-ul laptopului (din ipconfig)")
            .setView(input)
            .setPositiveButton("Salveaza", (dialog, which) -> {
                String ip = input.getText().toString().trim();
                if (!ip.isEmpty()) {
                    ApiClient.saveServerIp(ip);
                    Toast.makeText(this, "IP salvat: " + ip, Toast.LENGTH_SHORT).show();
                }
            })
            .setNegativeButton("Anuleaza", null)
            .show();
    }

    protected void logout() {
        ApiClient.clearToken();
        Intent intent = new Intent(this, LoginActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(intent);
    }
}
