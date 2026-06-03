package com.photomatch.db;

import android.database.Cursor;
import androidx.annotation.NonNull;
import androidx.room.EntityDeletionOrUpdateAdapter;
import androidx.room.EntityInsertionAdapter;
import androidx.room.RoomDatabase;
import androidx.room.RoomSQLiteQuery;
import androidx.room.util.CursorUtil;
import androidx.room.util.DBUtil;
import androidx.sqlite.db.SupportSQLiteStatement;
import java.lang.Class;
import java.lang.Override;
import java.lang.String;
import java.lang.SuppressWarnings;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.annotation.processing.Generated;

@Generated("androidx.room.RoomProcessor")
@SuppressWarnings({"unchecked", "deprecation"})
public final class FavoriteDao_Impl implements FavoriteDao {
  private final RoomDatabase __db;

  private final EntityInsertionAdapter<FavoritePhoto> __insertionAdapterOfFavoritePhoto;

  private final EntityDeletionOrUpdateAdapter<FavoritePhoto> __deletionAdapterOfFavoritePhoto;

  public FavoriteDao_Impl(@NonNull final RoomDatabase __db) {
    this.__db = __db;
    this.__insertionAdapterOfFavoritePhoto = new EntityInsertionAdapter<FavoritePhoto>(__db) {
      @Override
      @NonNull
      protected String createQuery() {
        return "INSERT OR ABORT INTO `favorites` (`id`,`originalBase64`,`correctedBase64`,`uriString`,`retrieved`,`timestamp`,`improvements`) VALUES (nullif(?, 0),?,?,?,?,?,?)";
      }

      @Override
      protected void bind(@NonNull final SupportSQLiteStatement statement,
          final FavoritePhoto entity) {
        statement.bindLong(1, entity.id);
        if (entity.originalBase64 == null) {
          statement.bindNull(2);
        } else {
          statement.bindString(2, entity.originalBase64);
        }
        if (entity.correctedBase64 == null) {
          statement.bindNull(3);
        } else {
          statement.bindString(3, entity.correctedBase64);
        }
        if (entity.uriString == null) {
          statement.bindNull(4);
        } else {
          statement.bindString(4, entity.uriString);
        }
        if (entity.retrieved == null) {
          statement.bindNull(5);
        } else {
          statement.bindString(5, entity.retrieved);
        }
        statement.bindLong(6, entity.timestamp);
        if (entity.improvements == null) {
          statement.bindNull(7);
        } else {
          statement.bindString(7, entity.improvements);
        }
      }
    };
    this.__deletionAdapterOfFavoritePhoto = new EntityDeletionOrUpdateAdapter<FavoritePhoto>(__db) {
      @Override
      @NonNull
      protected String createQuery() {
        return "DELETE FROM `favorites` WHERE `id` = ?";
      }

      @Override
      protected void bind(@NonNull final SupportSQLiteStatement statement,
          final FavoritePhoto entity) {
        statement.bindLong(1, entity.id);
      }
    };
  }

  @Override
  public void insert(final FavoritePhoto f) {
    __db.assertNotSuspendingTransaction();
    __db.beginTransaction();
    try {
      __insertionAdapterOfFavoritePhoto.insert(f);
      __db.setTransactionSuccessful();
    } finally {
      __db.endTransaction();
    }
  }

  @Override
  public void delete(final FavoritePhoto f) {
    __db.assertNotSuspendingTransaction();
    __db.beginTransaction();
    try {
      __deletionAdapterOfFavoritePhoto.handle(f);
      __db.setTransactionSuccessful();
    } finally {
      __db.endTransaction();
    }
  }

  @Override
  public List<FavoritePhoto> getAll() {
    final String _sql = "SELECT * FROM favorites ORDER BY timestamp DESC";
    final RoomSQLiteQuery _statement = RoomSQLiteQuery.acquire(_sql, 0);
    __db.assertNotSuspendingTransaction();
    final Cursor _cursor = DBUtil.query(__db, _statement, false, null);
    try {
      final int _cursorIndexOfId = CursorUtil.getColumnIndexOrThrow(_cursor, "id");
      final int _cursorIndexOfOriginalBase64 = CursorUtil.getColumnIndexOrThrow(_cursor, "originalBase64");
      final int _cursorIndexOfCorrectedBase64 = CursorUtil.getColumnIndexOrThrow(_cursor, "correctedBase64");
      final int _cursorIndexOfUriString = CursorUtil.getColumnIndexOrThrow(_cursor, "uriString");
      final int _cursorIndexOfRetrieved = CursorUtil.getColumnIndexOrThrow(_cursor, "retrieved");
      final int _cursorIndexOfTimestamp = CursorUtil.getColumnIndexOrThrow(_cursor, "timestamp");
      final int _cursorIndexOfImprovements = CursorUtil.getColumnIndexOrThrow(_cursor, "improvements");
      final List<FavoritePhoto> _result = new ArrayList<FavoritePhoto>(_cursor.getCount());
      while (_cursor.moveToNext()) {
        final FavoritePhoto _item;
        _item = new FavoritePhoto();
        _item.id = _cursor.getInt(_cursorIndexOfId);
        if (_cursor.isNull(_cursorIndexOfOriginalBase64)) {
          _item.originalBase64 = null;
        } else {
          _item.originalBase64 = _cursor.getString(_cursorIndexOfOriginalBase64);
        }
        if (_cursor.isNull(_cursorIndexOfCorrectedBase64)) {
          _item.correctedBase64 = null;
        } else {
          _item.correctedBase64 = _cursor.getString(_cursorIndexOfCorrectedBase64);
        }
        if (_cursor.isNull(_cursorIndexOfUriString)) {
          _item.uriString = null;
        } else {
          _item.uriString = _cursor.getString(_cursorIndexOfUriString);
        }
        if (_cursor.isNull(_cursorIndexOfRetrieved)) {
          _item.retrieved = null;
        } else {
          _item.retrieved = _cursor.getString(_cursorIndexOfRetrieved);
        }
        _item.timestamp = _cursor.getLong(_cursorIndexOfTimestamp);
        if (_cursor.isNull(_cursorIndexOfImprovements)) {
          _item.improvements = null;
        } else {
          _item.improvements = _cursor.getString(_cursorIndexOfImprovements);
        }
        _result.add(_item);
      }
      return _result;
    } finally {
      _cursor.close();
      _statement.release();
    }
  }

  @Override
  public FavoritePhoto findByRetrieved(final String retrieved) {
    final String _sql = "SELECT * FROM favorites WHERE retrieved = ? LIMIT 1";
    final RoomSQLiteQuery _statement = RoomSQLiteQuery.acquire(_sql, 1);
    int _argIndex = 1;
    if (retrieved == null) {
      _statement.bindNull(_argIndex);
    } else {
      _statement.bindString(_argIndex, retrieved);
    }
    __db.assertNotSuspendingTransaction();
    final Cursor _cursor = DBUtil.query(__db, _statement, false, null);
    try {
      final int _cursorIndexOfId = CursorUtil.getColumnIndexOrThrow(_cursor, "id");
      final int _cursorIndexOfOriginalBase64 = CursorUtil.getColumnIndexOrThrow(_cursor, "originalBase64");
      final int _cursorIndexOfCorrectedBase64 = CursorUtil.getColumnIndexOrThrow(_cursor, "correctedBase64");
      final int _cursorIndexOfUriString = CursorUtil.getColumnIndexOrThrow(_cursor, "uriString");
      final int _cursorIndexOfRetrieved = CursorUtil.getColumnIndexOrThrow(_cursor, "retrieved");
      final int _cursorIndexOfTimestamp = CursorUtil.getColumnIndexOrThrow(_cursor, "timestamp");
      final int _cursorIndexOfImprovements = CursorUtil.getColumnIndexOrThrow(_cursor, "improvements");
      final FavoritePhoto _result;
      if (_cursor.moveToFirst()) {
        _result = new FavoritePhoto();
        _result.id = _cursor.getInt(_cursorIndexOfId);
        if (_cursor.isNull(_cursorIndexOfOriginalBase64)) {
          _result.originalBase64 = null;
        } else {
          _result.originalBase64 = _cursor.getString(_cursorIndexOfOriginalBase64);
        }
        if (_cursor.isNull(_cursorIndexOfCorrectedBase64)) {
          _result.correctedBase64 = null;
        } else {
          _result.correctedBase64 = _cursor.getString(_cursorIndexOfCorrectedBase64);
        }
        if (_cursor.isNull(_cursorIndexOfUriString)) {
          _result.uriString = null;
        } else {
          _result.uriString = _cursor.getString(_cursorIndexOfUriString);
        }
        if (_cursor.isNull(_cursorIndexOfRetrieved)) {
          _result.retrieved = null;
        } else {
          _result.retrieved = _cursor.getString(_cursorIndexOfRetrieved);
        }
        _result.timestamp = _cursor.getLong(_cursorIndexOfTimestamp);
        if (_cursor.isNull(_cursorIndexOfImprovements)) {
          _result.improvements = null;
        } else {
          _result.improvements = _cursor.getString(_cursorIndexOfImprovements);
        }
      } else {
        _result = null;
      }
      return _result;
    } finally {
      _cursor.close();
      _statement.release();
    }
  }

  @NonNull
  public static List<Class<?>> getRequiredConverters() {
    return Collections.emptyList();
  }
}
