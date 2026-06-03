package com.photomatch.db;

import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.Query;

import java.util.List;

@Dao
public interface FavoriteDao {

    @Insert
    void insert(FavoritePhoto f);

    @Delete
    void delete(FavoritePhoto f);

    @Query("SELECT * FROM favorites ORDER BY timestamp DESC")
    List<FavoritePhoto> getAll();

    @Query("SELECT * FROM favorites WHERE retrieved = :retrieved LIMIT 1")
    FavoritePhoto findByRetrieved(String retrieved);
}
