"""Letterboxd sync persistence helpers."""
import re
from typing import Optional, List
from database import get_db_connection

def normalize_lb_uri(uri: str) -> str:
    """
    Standardizes Letterboxd URIs to prevent duplicates.
    Example: https://letterboxd.com/username/film/slug/ -> https://letterboxd.com/film/slug/
    """
    if not uri or not isinstance(uri, str):
        return uri
    # Remove username component if present
    match = re.search(r'letterboxd\.com/[^/]+/film/([^/]+)/?', uri)
    if match:
        slug = match.group(1)
        return f"https://letterboxd.com/film/{slug}/"
    # Ensure it starts with https and has trailing slash
    uri = uri.strip()
    if not uri.endswith('/'):
        uri += '/'
    return uri

def _process_movie_db_sync(movie: dict, user_id: str, tmdb_id: Optional[int], poster_path: Optional[str]):
    """Synchronous DB logic for processing a movie."""
    conn = get_db_connection()
    media_type = 'movie'
    if not conn:
        return tmdb_id, poster_path, media_type, False
    try:
        with conn.cursor() as cur:
            # Check mapping
            if not tmdb_id and movie.get('letterboxd_uri'):
                cur.execute("SELECT tmdb_id, poster_path, media_type FROM letterboxd_mappings WHERE letterboxd_url = %s", (movie['letterboxd_uri'],))
                row = cur.fetchone()
                if row:
                    tmdb_id, poster_path, media_type = row

            # Check existing rating
            if tmdb_id:
                interaction = movie.get('interaction_type', 'watched')
                watched_date = movie.get('watched_date')

                if watched_date:
                    cur.execute("SELECT 1 FROM user_ratings WHERE user_id = %s AND tmdb_id = %s AND interaction_type = %s AND watched_date = %s", (user_id, tmdb_id, interaction, watched_date))
                else:
                    cur.execute("SELECT 1 FROM user_ratings WHERE user_id = %s AND tmdb_id = %s AND interaction_type = %s AND watched_date IS NULL", (user_id, tmdb_id, interaction))

                if cur.fetchone():
                    return tmdb_id, poster_path, media_type, True # Found exact record, skip

        return tmdb_id, poster_path, media_type, False
    finally:
        conn.close()

def _save_movies_batch(results: List[tuple], user_id: str):
    """Saves a batch of movies to the database in a single transaction."""
    conn = get_db_connection()
    if not conn:
        print(f"[SYNC][ERROR] DB connection failed for user {user_id}")
        return
    try:
        with conn.cursor() as cur:
            for movie, tmdb_id, poster_path, media_type in results:
                # 1. Update Mapping Cache
                normalized_uri = normalize_lb_uri(movie['letterboxd_uri'])
                cur.execute("INSERT INTO letterboxd_mappings (letterboxd_url, tmdb_id, poster_path, media_type) VALUES (%s, %s, %s, %s) ON CONFLICT (letterboxd_url) DO UPDATE SET tmdb_id = EXCLUDED.tmdb_id, poster_path = EXCLUDED.poster_path, media_type = EXCLUDED.media_type", (normalized_uri, tmdb_id, poster_path, media_type))

                # 2. Save User Rating/Watchlist Item
                interaction = movie.get('interaction_type', 'watched')
                cur.execute("""
                    INSERT INTO user_ratings (user_id, movie_title, release_year, rating, watched_date, is_liked, interaction_type, tmdb_id, poster_path, media_type, letterboxd_uri, letterboxd_watch_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, letterboxd_uri)
                    DO UPDATE SET
                        rating = EXCLUDED.rating,
                        watched_date = EXCLUDED.watched_date,
                        is_liked = EXCLUDED.is_liked,
                        tmdb_id = COALESCE(EXCLUDED.tmdb_id, user_ratings.tmdb_id),
                        poster_path = COALESCE(EXCLUDED.poster_path, user_ratings.poster_path),
                        interaction_type = EXCLUDED.interaction_type,
                        media_type = EXCLUDED.media_type,
                        letterboxd_watch_id = COALESCE(EXCLUDED.letterboxd_watch_id, user_ratings.letterboxd_watch_id)
                """, (user_id, movie['movie_title'], movie.get('release_year'), movie.get('rating'), movie.get('watched_date'), movie.get('is_liked', False), interaction, tmdb_id, poster_path, media_type, normalized_uri, movie.get('letterboxd_watch_id')))

                # 3. CLEANUP: If this is a 'watched' entry, remove it from 'watchlist'
                if interaction == 'watched' and tmdb_id:
                    cur.execute("DELETE FROM user_ratings WHERE user_id = %s AND tmdb_id = %s AND interaction_type = 'watchlist'", (user_id, tmdb_id))
            conn.commit()
            print(f"[SYNC][BATCH] Successfully committed {len(results)} movies.")
    finally:
        conn.close()

def _cleanup_watchlist_duplicates(user_id: str):
    """Removes watchlist entries for movies that have a corresponding watched/rated entry."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM user_ratings
                WHERE user_id = %s
                AND interaction_type = 'watchlist'
                AND tmdb_id IN (
                    SELECT tmdb_id FROM user_ratings
                    WHERE user_id = %s AND interaction_type = 'watched'
                )
            """, (user_id, user_id))
            count = cur.rowcount
            conn.commit()
            if count > 0:
                print(f"[SYNC][CLEANUP] Purged {count} redundant watchlist items for user {user_id}.")
    finally:
        conn.close()
