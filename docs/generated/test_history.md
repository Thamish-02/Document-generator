## AI Summary

A file named test_history.py.


### Function: test_proper_default_encoding()

### Function: test_history()

### Function: test_extract_hist_ranges()

### Function: test_extract_hist_ranges_empty_str()

### Function: test_magic_rerun()

**Description:** Simple test for %rerun (no args -> rerun last line)

### Function: test_timestamp_type()

### Function: test_hist_file_config()

### Function: test_histmanager_disabled()

**Description:** Ensure that disabling the history manager doesn't create a database.

### Function: test_get_tail_session_awareness()

**Description:** Test .get_tail() is:
    - session specific in HistoryManager
    - session agnostic in HistoryAccessor
same for .get_last_session_id()
