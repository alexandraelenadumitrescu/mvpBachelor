package com.photomatch.db;

import android.content.Context;

import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;

@Database(entities = {FavoritePhoto.class}, version = 1, exportSchema = false)
public abstract class AppDatabase extends RoomDatabase {

    private static volatile AppDatabase INSTANCE;

    public abstract FavoriteDao favoriteDao();

    public static AppDatabase get(Context ctx) {
        if (INSTANCE == null) {
            synchronized (AppDatabase.class) {
                if (INSTANCE == null) {
                    INSTANCE = Room.databaseBuilder(
                        ctx.getApplicationContext(),
                        AppDatabase.class,
                        "photomatch.db"
                    ).build();
                }
            }
        }
        return INSTANCE;
    }
}
