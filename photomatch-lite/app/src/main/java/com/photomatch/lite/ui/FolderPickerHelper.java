package com.photomatch.lite.ui;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import androidx.documentfile.provider.DocumentFile;
import java.util.ArrayList;
import java.util.List;

public class FolderPickerHelper {

    private static final java.util.Set<String> IMAGE_TYPES = new java.util.HashSet<>(
        java.util.Arrays.asList("image/jpeg", "image/jpg", "image/png", "image/webp")
    );

    public static List<Uri> listImages(Context context, Uri treeUri) {
        try {
            context.getContentResolver().takePersistableUriPermission(
                treeUri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (Exception ignored) {}

        List<Uri> result = new ArrayList<>();
        DocumentFile folder = DocumentFile.fromTreeUri(context, treeUri);
        if (folder == null) return result;

        for (DocumentFile file : folder.listFiles()) {
            if (file.isFile() && IMAGE_TYPES.contains(file.getType())) {
                result.add(file.getUri());
            }
        }
        return result;
    }
}
