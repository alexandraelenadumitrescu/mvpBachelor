package com.photomatch.db;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "favorites")
public class FavoritePhoto {
    @PrimaryKey(autoGenerate = true)
    public int    id;
    public String originalBase64;   // resp.originalB64 (null for burst)
    public String correctedBase64;  // resp.finalB64 (null for burst)
    public String uriString;        // content URI for burst photos (null for server results)
    public String retrieved;        // resp.retrieved or URI string — used for dedup lookup
    public long   timestamp;
    public String improvements;     // JSON map of defect scores + metadata
}
