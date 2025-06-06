# test_media_db_v2_properties.py
#
# Property-based tests for the Media_DB_v2 library using Hypothesis.
# These tests verify the logical correctness and invariants of the database
# operations across a wide range of generated data.
#
# Imports
from datetime import datetime, timezone, timedelta
from typing import Iterator
import pytest
import uuid
from pathlib import Path
#
# Third-Party Imports
from hypothesis import given, strategies as st, settings, HealthCheck, assume
#
# Local Imports
# Adjust the import path based on your project structure
from tldw_chatbook.DB.Client_Media_DB_v2 import (
    MediaDatabase,
    InputError,
    DatabaseError,
    ConflictError, fetch_keywords_for_media, empty_trash
)
#
#######################################################################################################################
#
# --- Hypothesis Settings ---

# A custom profile for database-intensive tests.
# It increases the deadline and suppresses health checks that are common
# but expected in I/O-heavy testing scenarios.
settings.register_profile(
    "db_test_suite",
    deadline=2000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.data_too_large,
    ]
)
settings.load_profile("db_test_suite")


# --- Pytest Fixtures ---

@pytest.fixture(scope="function")
def db_instance(tmp_path: Path) -> Iterator[MediaDatabase]:
    """
    Creates a fresh, on-disk MediaDatabase instance for each test function.
    The database file is created in a temporary directory that pytest manages
    and automatically cleans up. This ensures complete test isolation.
    """
    db_file = tmp_path / f"prop_test_{uuid.uuid4().hex}.db"
    client_id = f"client_{uuid.uuid4().hex[:8]}"

    # The Database class handles directory creation.
    db = MediaDatabase(db_path=db_file, client_id=client_id)

    yield db

    # Teardown: close connection before pytest deletes the temp directory.
    db.close_connection()


# --- Hypothesis Strategies ---

# Strategy for text that is safe to use in FTS queries (no special chars).
st_searchable_text = st.text(
    alphabet=st.characters(whitelist_categories=["L", "N", "P", "S", "Cn"], min_codepoint=32, max_codepoint=122),
    min_size=3
).map(lambda s: s.strip()).filter(lambda s: len(s) > 0)

# Strategy for generating text that is guaranteed to have non-whitespace content.
st_required_text = st.from_regex(r".*\S.*", fullmatch=True).map(lambda s: s.strip())

# Strategy for a single, clean keyword.
st_keyword_text = st.text(
    alphabet=st.characters(whitelist_categories=["L", "N", "P", "S", "Cn"]),
    min_size=2,
    max_size=20
)

# Strategy for generating a list of unique, case-insensitive keywords.
# This reuses the clean `st_keyword_text` strategy.
st_keywords_list = st.lists(
    st_keyword_text,
    min_size=1,
    max_size=5,
    unique_by=lambda s: s.lower()
)

# A composite strategy to generate a valid dictionary of media data for creation.
@st.composite
def st_media_data(draw):
    """Generates a dictionary of plausible data for a new media item."""
    return {
        "title": draw(st_searchable_text), # FIX: Use searchable text for titles
        "content": draw(st.text(min_size=10, max_size=500)),
        "media_type": draw(st.sampled_from(['article', 'video', 'obsidian_note', 'pdf'])),
        "author": draw(st.one_of(st.none(), st.text(min_size=3, max_size=30))),
        "keywords": draw(st_keywords_list)
    }


# --- Property Test Classes ---

class TestMediaItemProperties:
    """Property-based tests for the core Media item lifecycle."""

    @given(media_data=st_media_data())
    def test_media_item_roundtrip(self, db_instance: MediaDatabase, media_data: dict):
        """
        Property: A media item, once added, should be retrievable with the same data.
        This is a "round-trip" test verifying the `add` and `get` operations.
        """
        # Ensure content is unique for this test run to prevent accidental collisions
        media_data["content"] += f" {uuid.uuid4().hex}"

        media_id, media_uuid, msg = db_instance.add_media_with_keywords(**media_data)

        assert "added" in msg
        assert media_id is not None
        assert media_uuid is not None

        retrieved = db_instance.get_media_by_id(media_id)
        assert retrieved is not None

        # Verify core data
        assert retrieved['title'] == media_data['title']
        assert retrieved['content'] == media_data['content']
        assert retrieved['media_type'] == media_data['media_type']
        assert retrieved['author'] == media_data['author']
        assert retrieved['version'] == 1
        assert not retrieved['deleted']

        # Verify linked keywords (case-insensitively)
        linked_keywords = {kw.lower().strip() for kw in fetch_keywords_for_media(media_id, db_instance)}
        expected_keywords = {kw.lower().strip() for kw in media_data['keywords']}
        assert linked_keywords == expected_keywords

        # Verify a DocumentVersion was created
        doc_versions = db_instance.get_all_document_versions(media_id)
        assert len(doc_versions) == 1
        assert doc_versions[0]['version_number'] == 1
        assert doc_versions[0]['content'] == media_data['content']

    @given(initial_media=st_media_data(), update_media=st_media_data())
    def test_update_increments_version_and_changes_data(self, db_instance: MediaDatabase, initial_media: dict, update_media: dict):
        """
        Property: Updating an existing media item must increment its version by exactly 1
        and correctly apply the new data.
        """
        # Ensure content is unique to prevent collisions during creation
        initial_media["content"] += f" initial_{uuid.uuid4().hex}"
        update_media["content"] += f" update_{uuid.uuid4().hex}"

        media_id, media_uuid, _ = db_instance.add_media_with_keywords(**initial_media)
        original = db_instance.get_media_by_id(media_id)

        # perform an update by providing the stable URL identifier
        media_id_up, media_uuid_up, msg = db_instance.add_media_with_keywords(
            url=original['url'],
            overwrite=True,
            **update_media
        )

        assert media_id_up == media_id
        assert media_uuid_up == media_uuid
        assert "updated" in msg

        updated = db_instance.get_media_by_id(media_id)
        assert updated is not None
        assert updated['version'] == original['version'] + 1
        assert updated['title'] == update_media['title']
        assert updated['content'] == update_media['content']

        # A new document version should be created for the update
        doc_versions = db_instance.get_all_document_versions(media_id)
        assert len(doc_versions) == 2

    @given(media_data=st_media_data())
    def test_soft_delete_makes_item_unfindable_by_default(self, db_instance: MediaDatabase, media_data: dict):
        """
        Property: A soft-deleted item must not be returned by standard `get` or `search`
        methods unless explicitly requested.
        """
        # FIX: Ensure content is unique
        media_data["content"] += f" {uuid.uuid4().hex}"

        media_id, _, _ = db_instance.add_media_with_keywords(**media_data)
        original = db_instance.get_media_by_id(media_id)
        assert original is not None

        # Perform soft delete
        db_instance.soft_delete_media(media_id)

        # Assert it's unfindable by default
        assert db_instance.get_media_by_id(media_id) is None
        results, total = db_instance.search_media_db(search_query=media_data['title'])
        assert total == 0

        # Assert it IS findable when requested
        assert db_instance.get_media_by_id(media_id, include_deleted=True) is not None
        results_del, total_del = db_instance.search_media_db(search_query=media_data['title'], include_deleted=True)
        assert total_del > 0

        # Assert its state in the DB is correct
        raw_record = db_instance.get_media_by_id(media_id, include_deleted=True)
        assert raw_record['deleted'] == 1
        assert raw_record['version'] == original['version'] + 1


class TestSearchProperties:
    """Property-based tests to verify the invariants of the search function."""

    @given(media_data=st_media_data())
    def test_search_finds_item_by_its_properties(self, db_instance: MediaDatabase, media_data: dict):
        """
        Invariant: An item added to the database must be findable by its own distinct properties.
        """
        # Make a piece of content unique to avoid collisions from other test data
        unique_word = f"hypothesis_{uuid.uuid4().hex}"
        media_data["content"] = f"{media_data['content']} {unique_word}"
        media_id, _, _ = db_instance.add_media_with_keywords(**media_data)

        # Invariant 1: Search by unique content word
        results, total = db_instance.search_media_db(search_query=unique_word, search_fields=['content'])
        assert total == 1
        assert results[0]['id'] == media_id

        # Invariant 2: Search by one of its keywords
        keyword_to_find = media_data["keywords"][0]
        # FIX: Search by keyword only, not title, to isolate the filter
        results, total = db_instance.search_media_db(search_query=None, must_have_keywords=[keyword_to_find])
        assert total == 1
        assert results[0]['id'] == media_id

        # FIX: Search by media type only
        results, total = db_instance.search_media_db(search_query=None, media_types=[media_data['media_type']])
        assert total >= 1
        assert any(r['id'] == media_id for r in results)

    @given(item1=st_media_data(), item2=st_media_data())
    def test_search_isolates_results_correctly(self, db_instance: MediaDatabase, item1: dict, item2: dict):
        """
        Invariant: A search for properties of item1 should not return item2 if their
        properties are distinct.
        """
        # Ensure the items have no overlapping keywords to make the test clean.
        # Hypothesis will try to find examples where this assumption holds.
        item1_kws = set(kw.lower() for kw in item1['keywords'])
        item2_kws = set(kw.lower() for kw in item2['keywords'])
        assume(item1_kws.isdisjoint(item2_kws))

        # FIX: Ensure content uniqueness
        item1["content"] += f" item1_{uuid.uuid4().hex}"
        item2["content"] += f" item2_{uuid.uuid4().hex}"

        id1, _, _ = db_instance.add_media_with_keywords(**item1)
        id2, _, _ = db_instance.add_media_with_keywords(**item2)

        # Search for a keyword that only item1 has
        keyword_to_find = item1['keywords'][0]
        # FIX: Search by keyword only
        results, total = db_instance.search_media_db(search_query=None, must_have_keywords=[keyword_to_find])

        # Assert that we found item1 and ONLY item1
        assert total == 1
        assert len(results) == 1
        assert results[0]['id'] == id1

    @given(media_data=st_media_data())
    def test_soft_deleted_item_is_not_in_fts_search(self, db_instance: MediaDatabase, media_data: dict):
        """
        Invariant: When an item is soft-deleted, its entry in the FTS table must
        be removed, making it unsearchable via FTS.
        """
        unique_term = f"fts_{uuid.uuid4().hex}"
        media_data['title'] = f"{media_data['title']} {unique_term}"
        # FIX: Ensure content uniqueness
        media_data['content'] += f" {uuid.uuid4().hex}"
        media_id, _, _ = db_instance.add_media_with_keywords(**media_data)

        # 1. Verify it's searchable before deletion
        results, total = db_instance.search_media_db(search_query=unique_term)
        assert total == 1

        # 2. Soft-delete the item
        db_instance.soft_delete_media(media_id)

        # 3. Verify it's no longer found by the FTS search
        results, total = db_instance.search_media_db(search_query=unique_term)
        assert total == 0


class TestSyncLogAndVersioning:
    """Property tests for the sync log and versioning system."""

    @given(media_data=st_media_data())
    def test_add_media_creates_correct_sync_logs(self, db_instance: MediaDatabase, media_data: dict):
        """
        Property: Creating a media item must generate 'create' logs for the Media item,
        its DocumentVersion, its Keywords, and 'link' logs for the associations.
        """
        # FIX: Ensure content uniqueness
        media_data["content"] += f" {uuid.uuid4().hex}"
        log_count_before = len(db_instance.get_sync_log_entries())

        # Action
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(**media_data)

        # Get all logs created during the operation
        new_logs = db_instance.get_sync_log_entries(since_change_id=log_count_before)

        log_operations = [log['operation'] for log in new_logs]
        log_entities = [log['entity'] for log in new_logs]

        # Assert a 'create' log for the Media item itself
        media_create_log = next((log for log in new_logs if log['entity'] == 'Media' and log['operation'] == 'create'), None)
        assert media_create_log is not None, "Media 'create' log was not found"
        assert media_create_log['entity_uuid'] == media_uuid
        assert media_create_log['version'] == 1

        # Assert a 'create' log for the DocumentVersion
        doc_version_log = next((log for log in new_logs if log['entity'] == 'DocumentVersions' and log['operation'] == 'create'), None)
        assert doc_version_log is not None, "DocumentVersion 'create' log was not found"

        # Assert 'link' logs for each keyword association
        link_logs = [log for log in new_logs if log['entity'] == 'MediaKeywords' and log['operation'] == 'link']
        assert len(link_logs) == len(set(kw.lower() for kw in media_data['keywords']))

    @given(media_data=st_media_data())
    def test_soft_delete_media_creates_delete_and_unlink_logs(self, db_instance: MediaDatabase, media_data: dict):
        """
        Property: Soft-deleting a media item must log a 'delete' event for the media
        and 'unlink' events for all its keyword associations.
        """
        # FIX: Ensure content uniqueness
        media_data["content"] += f" {uuid.uuid4().hex}"

        media_id, media_uuid, _ = db_instance.add_media_with_keywords(**media_data)
        log_count_before = len(db_instance.get_sync_log_entries())

        # Action
        db_instance.soft_delete_media(media_id)

        # Get logs created by the delete operation
        new_logs = db_instance.get_sync_log_entries(since_change_id=log_count_before)

        # Assert 'delete' log for the Media item
        media_delete_log = next((log for log in new_logs if log['entity'] == 'Media' and log['operation'] == 'delete'), None)
        assert media_delete_log is not None
        assert media_delete_log['entity_uuid'] == media_uuid
        assert media_delete_log['version'] == 2 # Version 1 was create, 2 is delete

        unlink_logs = [log for log in new_logs if log['entity'] == 'MediaKeywords' and log['operation'] == 'unlink']
        assert len(unlink_logs) == len(set(kw.lower() for kw in media_data['keywords']))


class TestConcurrencyAndIntegrity:
    """Tests for database integrity constraints and multi-threading behavior."""

    def test_each_thread_gets_a_separate_connection(self, db_instance: MediaDatabase):
        """
        Property: The connection management must provide a unique connection
        object for each thread, as required by SQLite.
        """
        import threading

        connection_ids = set()
        lock = threading.Lock()
        errors = []

        def get_and_store_conn_id():
            try:
                conn = db_instance.get_connection()
                with lock:
                    connection_ids.add(id(conn))
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=get_and_store_conn_id) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # If threading.local is working correctly, each thread gets its own connection.
        assert len(connection_ids) == 5

    def test_add_media_with_duplicate_url_is_handled(self, db_instance: MediaDatabase):
        """
        Property: Attempting to add a new media item with a URL that already
        exists must be handled correctly (skipped by default, updated if overwrite=True).
        """
        url = f"http://example.com/unique-article-{uuid.uuid4()}"
        # FIX: Add missing media_type to test data
        media1_data = {"title": "First", "content": "Content A", "url": url, "media_type": "article"}
        media2_data = {"title": "Second", "content": "Content B", "url": url, "media_type": "article"}

        # 1. Add the first item
        id1, uuid1, msg1 = db_instance.add_media_with_keywords(**media1_data)
        assert "added" in msg1

        # 2. Try to add the second with the same URL (overwrite=False)
        id2, uuid2, msg2 = db_instance.add_media_with_keywords(**media2_data, overwrite=False)
        assert "already_exists_skipped" in msg2
        assert id2 is None # Should not return an ID if skipped

        # 3. Add the second with overwrite=True
        id3, uuid3, msg3 = db_instance.add_media_with_keywords(**media2_data, overwrite=True)
        assert "updated" in msg3
        assert id3 == id1 # Should have updated the original item
        assert uuid3 == uuid1

        # Verify the content was updated
        final_item = db_instance.get_media_by_id(id1)
        assert final_item['content'] == "Content B"
        assert final_item['version'] == 2


class TestStateTransitionsAndComplexOps:
    """Tests for complex operations and transitions between different states."""

    @given(initial_data=st_media_data(), update_data=st_media_data())
    def test_rollback_creates_new_version_with_old_content(self, db_instance: MediaDatabase, initial_data: dict,
                                                           update_data: dict):
        """
        Property: Rolling back to a previous document version must:
        1. Update the main Media item's content to match the old version's content.
        2. Increment the Media item's own sync version.
        3. Create a *new* DocumentVersion, increasing the total version count.
        """
        # FIX: Ensure unique content to avoid test failures
        initial_data['content'] += f" initial_{uuid.uuid4().hex}"
        update_data['content'] += f" update_{uuid.uuid4().hex}"

        media_id, _, _ = db_instance.add_media_with_keywords(**initial_data)

        # 2. Update the media to create a second state (Document Version 2)
        media_item_v1 = db_instance.get_media_by_id(media_id)
        db_instance.add_media_with_keywords(url=media_item_v1['url'], overwrite=True, **update_data)

        media_item_v2 = db_instance.get_media_by_id(media_id)
        assert media_item_v2['version'] == 2
        versions_before_rollback = db_instance.get_all_document_versions(media_id)
        assert len(versions_before_rollback) == 2

        # 3. Perform the rollback to the first document version (version_number=1)
        rollback_result = db_instance.rollback_to_version(media_id, target_version_number=1)
        assert 'success' in rollback_result

        # 4. Verify the final state
        final_media_item = db_instance.get_media_by_id(media_id)
        final_versions = db_instance.get_all_document_versions(media_id)

        # The media item's content should now match the initial content
        assert final_media_item['content'] == initial_data['content']
        # The media item's sync version should have incremented again
        assert final_media_item['version'] == 3
        # There should now be a third document version
        assert len(final_versions) == 3
        # The newest document version should have the rolled-back content
        assert final_versions[0]['content'] == initial_data['content']

    @given(initial_keywords=st_keywords_list, final_keywords=st_keywords_list)
    def test_update_keywords_synchronizes_links(self, db_instance: MediaDatabase, initial_keywords: list,
                                                final_keywords: list):
        """
        Property: `update_keywords_for_media` must result in the media item being
        linked to *exactly* the set of `final_keywords`.
        """
        media_id, _, _ = db_instance.add_media_with_keywords(
            title="Keyword Sync Test", content=f"...{uuid.uuid4().hex}", keywords=initial_keywords, media_type="article")

        # Action: Synchronize keywords to the final list
        db_instance.update_keywords_for_media(media_id, final_keywords)

        # Verification
        linked_keywords = fetch_keywords_for_media(media_id, db_instance)

        # Compare the sets of lowercased keywords
        linked_set = {kw.lower() for kw in linked_keywords}
        expected_set = {kw.lower() for kw in final_keywords}

        assert linked_set == expected_set

    @given(media_data=st_media_data())
    def test_trash_and_delete_interactions(self, db_instance: MediaDatabase, media_data: dict):
        """
        Property: An item that is marked as trash and then soft-deleted should
        have both `is_trash=1` and `deleted=1`.
        """
        # FIX: Ensure content uniqueness
        media_data["content"] += f" {uuid.uuid4().hex}"
        media_id, _, _ = db_instance.add_media_with_keywords(**media_data)

        # 1. Mark as trash
        db_instance.mark_as_trash(media_id)
        item_in_trash = db_instance.get_media_by_id(media_id, include_trash=True)
        assert item_in_trash['is_trash'] == 1
        assert item_in_trash['deleted'] == 0

        # 2. Soft delete the trashed item
        db_instance.soft_delete_media(media_id)

        # 3. Verify final state
        final_item = db_instance.get_media_by_id(media_id, include_trash=True, include_deleted=True)
        assert final_item is not None
        assert final_item['is_trash'] == 1
        assert final_item['deleted'] == 1

        # It should not be findable by default search
        _, total = db_instance.search_media_db(search_query=media_data['title'])
        assert total == 0


class TestIdempotencyAndConstraints:
    """Tests for idempotency of operations and enforcement of DB constraints."""

    @given(media_data=st_media_data())
    def test_mark_as_trash_is_idempotent(self, db_instance: MediaDatabase, media_data: dict):
        """
        Property: Marking an item as trash multiple times has the same effect as
        marking it once. The version should only increment on the first call.
        """
        # FIX: Ensure content uniqueness
        media_data["content"] += f" {uuid.uuid4().hex}"
        media_id, _, _ = db_instance.add_media_with_keywords(**media_data)

        # First call should succeed and return True
        assert db_instance.mark_as_trash(media_id) is True
        item_v2 = db_instance.get_media_by_id(media_id, include_trash=True)
        assert item_v2['version'] == 2

        # Second call should do nothing and return False
        assert db_instance.mark_as_trash(media_id) is False
        item_still_v2 = db_instance.get_media_by_id(media_id, include_trash=True)
        assert item_still_v2['version'] == 2

    @given(media1=st_media_data(), media2=st_media_data())
    def test_add_media_with_conflicting_hash_is_handled(self, db_instance: MediaDatabase, media1: dict, media2: dict):
        """
        Property: The UNIQUE constraint on `content_hash` is handled gracefully.
        Adding an item with the same content as an existing item should be
        skipped or updated, not raise an IntegrityError.
        """
        # FIX: Ensure titles are different to isolate the test to content hash collision
        assume(media1['title'] != media2['title'])
        media2['content'] = media1['content']

        # Add the first item
        id1, uuid1, msg1 = db_instance.add_media_with_keywords(**media1)
        assert "added" in msg1

        # Try to add the second item with the same content (overwrite=False)
        id2, uuid2, msg2 = db_instance.add_media_with_keywords(**media2, overwrite=False)
        assert "skipped" in msg2
        assert id2 is None

        # Now try with overwrite=True
        id3, uuid3, msg3 = db_instance.add_media_with_keywords(**media2, overwrite=True)
        assert "updated" in msg3
        assert id3 == id1

        final_item = db_instance.get_media_by_id(id1)
        assert final_item['title'] == media2['title']
        assert final_item['version'] == 2


class TestTimeBasedAndSearchQueries:
    """Tests involving time-based logic and search query combinations."""

    # FIX: Correct the datetime strategy
    @given(media_data=st_media_data(),
           ingestion_dt_naive=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1)))
    def test_search_by_date_range_finds_item(self, db_instance: MediaDatabase, media_data: dict,
                                             ingestion_dt_naive: datetime):
        """
        Invariant: An item added with a specific ingestion date must be found
        when searching a date range that includes that date.
        """
        # FIX: Make the datetime timezone-aware after generation
        ingestion_dt = ingestion_dt_naive.replace(tzinfo=timezone.utc)

        media_data['ingestion_date'] = ingestion_dt.isoformat()
        media_data['content'] += f" {uuid.uuid4().hex}" # Ensure unique
        media_id, _, _ = db_instance.add_media_with_keywords(**media_data)

        # 1. Search a range that INCLUDES the date
        search_range_inclusive = {
            'start_date': ingestion_dt - timedelta(days=1),
            'end_date': ingestion_dt + timedelta(days=1)
        }
        # FIX: Search by a unique property to avoid test pollution
        results, total = db_instance.search_media_db(date_range=search_range_inclusive, media_ids_filter=[media_id], search_query=media_data['content'])
        assert total == 1
        assert results[0]['id'] == media_id

        # 2. Search a range that EXCLUDES the date (in the future)
        search_range_future = {
            'start_date': ingestion_dt + timedelta(days=1),
            'end_date': ingestion_dt + timedelta(days=2)
        }
        results, total = db_instance.search_media_db(date_range=search_range_future, media_ids_filter=[media_id], search_query=media_data['content'])
        assert total == 0

    @given(days=st.integers(min_value=1, max_value=365))
    def test_empty_trash_respects_time_threshold(self, db_instance: MediaDatabase, days: int):
        """
        Property: `empty_trash` should only soft-delete items whose `trash_date`
        is older than the specified threshold.
        """
        media_id, _, _ = db_instance.add_media_with_keywords(
            title="Trash Test", content=f"...{uuid.uuid4().hex}", media_type="article", keywords=["test"])

        db_instance.mark_as_trash(media_id)
        item_v2 = db_instance.get_media_by_id(media_id, include_trash=True)

        # 2. Manually set its trash_date to be in the past.
        # This is a targeted way to test time-based logic without mocking `datetime`.
        past_date = datetime.now(timezone.utc) - timedelta(days=days + 1)

        # FIX: Manually update the row while respecting the versioning trigger.
        db_instance.execute_query(
            "UPDATE Media SET trash_date = ?, version = ?, last_modified = ?, client_id = ? WHERE id = ?",
            (
                past_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                item_v2['version'] + 1,
                db_instance._get_current_utc_timestamp_str(),
                'test_manual_update_client',
                media_id
            ),
            commit=True
        )

        processed_count, _ = empty_trash(db_instance=db_instance, days_threshold=days)
        assert processed_count == 1

        # 4. Verify the item is now soft-deleted
        final_item = db_instance.get_media_by_id(media_id, include_trash=True, include_deleted=True)
        assert final_item['deleted'] == 1
        assert final_item['version'] == item_v2['version'] + 2 # V2 from trash, V3 from manual update, V4 from soft_delete


#
# End of test_media_db_properties.py
#######################################################################################################################
