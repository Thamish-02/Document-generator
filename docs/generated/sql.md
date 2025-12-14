## AI Summary

A file named sql.py.


### Function: _compile_varchar_mysql(element, compiler)

**Description:** MySQL hack to avoid the "VARCHAR requires a length" error.

## Class: _SQLitePatch

**Description:** Used internally by L{BaseDAO}.

After connecting to an SQLite database, ensure that the foreign keys
support is enabled. If not, abort the connection.

@see: U{http://sqlite.org/foreignkeys.html}

## Class: BaseDTO

**Description:** Customized declarative base for SQLAlchemy.

## Class: BaseDAO

**Description:** Data Access Object base class.

@type _url: sqlalchemy.url.URL
@ivar _url: Database connection URL.

@type _dialect: str
@ivar _dialect: SQL dialect currently being used.

@type _driver: str
@ivar _driver: Name of the database driver currently being used.
    To get the actual Python module use L{_url}.get_driver() instead.

@type _session: sqlalchemy.orm.Session
@ivar _session: Database session object.

@type _new_session: class
@cvar _new_session: Custom configured Session class used to create the
    L{_session} instance variable.

@type _echo: bool
@cvar _echo: Set to C{True} to print all SQL queries to standard output.

### Function: Transactional(fn, self)

**Description:** Decorator that wraps DAO methods to handle transactions automatically.

It may only work with subclasses of L{BaseDAO}.

### Function: _gen_valid_access_flags()

## Class: MemoryDTO

**Description:** Database mapping for memory dumps.

## Class: CrashDTO

**Description:** Database mapping for crash dumps.

## Class: CrashDAO

**Description:** Data Access Object to read, write and search for L{Crash} objects in a
database.

### Function: connect(dbapi_connection, connection_record)

**Description:** Called once by SQLAlchemy for each new SQLite DB-API connection.

Here is where we issue some PRAGMA statements to configure how we're
going to access the SQLite database.

@param dbapi_connection:
    A newly connected raw SQLite DB-API connection.

@param connection_record:
    Unused by this method.

### Function: __init__(self, url, creator)

**Description:** Connect to the database using the given connection URL.

The current implementation uses SQLAlchemy and so it will support
whatever database said module supports.

@type  url: str
@param url:
    URL that specifies the database to connect to.

    Some examples:
     - Opening an SQLite file:
       C{dao = CrashDAO("sqlite:///C:\some\path\database.sqlite")}
     - Connecting to a locally installed SQL Express database:
       C{dao = CrashDAO("mssql://.\SQLEXPRESS/Crashes?trusted_connection=yes")}
     - Connecting to a MySQL database running locally, using the
       C{oursql} library, authenticating as the "winappdbg" user with
       no password:
       C{dao = CrashDAO("mysql+oursql://winappdbg@localhost/Crashes")}
     - Connecting to a Postgres database running locally,
       authenticating with user and password:
       C{dao = CrashDAO("postgresql://winappdbg:winappdbg@localhost/Crashes")}

    For more information see the C{SQLAlchemy} documentation online:
    U{http://docs.sqlalchemy.org/en/latest/core/engines.html}

    Note that in all dialects except for SQLite the database
    must already exist. The tables schema, however, is created
    automatically when connecting for the first time.

    To create the database in MSSQL, you can use the
    U{SQLCMD<http://msdn.microsoft.com/en-us/library/ms180944.aspx>}
    command::
        sqlcmd -Q "CREATE DATABASE Crashes"

    In MySQL you can use something like the following::
        mysql -u root -e "CREATE DATABASE Crashes;"

    And in Postgres::
        createdb Crashes -h localhost -U winappdbg -p winappdbg -O winappdbg

    Some small changes to the schema may be tolerated (for example,
    increasing the maximum length of string columns, or adding new
    columns with default values). Of course, it's best to test it
    first before making changes in a live database. This all depends
    very much on the SQLAlchemy version you're using, but it's best
    to use the latest version always.

@type  creator: callable
@param creator: (Optional) Callback function that creates the SQL
    database connection.

    Normally it's not necessary to use this argument. However in some
    odd cases you may need to customize the database connection.

### Function: _transactional(self, method)

**Description:** Begins a transaction and calls the given DAO method.

If the method executes successfully the transaction is commited.

If the method fails, the transaction is rolled back.

@type  method: callable
@param method: Bound method of this class or one of its subclasses.
    The first argument will always be C{self}.

@return: The return value of the method call.

@raise Exception: Any exception raised by the method.

### Function: __init__(self, crash_id, mbi)

**Description:** Process a L{win32.MemoryBasicInformation} object for database storage.

### Function: _to_access(self, protect)

### Function: toMBI(self, getMemoryDump)

**Description:** Returns a L{win32.MemoryBasicInformation} object using the data
retrieved from the database.

@type  getMemoryDump: bool
@param getMemoryDump: (Optional) If C{True} retrieve the memory dump.
    Defaults to C{False} since this may be a costly operation.

@rtype:  L{win32.MemoryBasicInformation}
@return: Memory block information.

### Function: _parse_state(state)

### Function: _parse_type(type)

### Function: _parse_access(access)

### Function: __init__(self, crash)

**Description:** @type  crash: Crash
@param crash: L{Crash} object to store into the database.

### Function: toCrash(self, getMemoryDump)

**Description:** Returns a L{Crash} object using the data retrieved from the database.

@type  getMemoryDump: bool
@param getMemoryDump: If C{True} retrieve the memory dump.
    Defaults to C{False} since this may be a costly operation.

@rtype:  L{Crash}
@return: Crash object.

### Function: add(self, crash, allow_duplicates)

**Description:** Add a new crash dump to the database, optionally filtering them by
signature to avoid duplicates.

@type  crash: L{Crash}
@param crash: Crash object.

@type  allow_duplicates: bool
@param allow_duplicates: (Optional)
    C{True} to always add the new crash dump.
    C{False} to only add the crash dump if no other crash with the
    same signature is found in the database.

    Sometimes, your fuzzer turns out to be I{too} good. Then you find
    youself browsing through gigabytes of crash dumps, only to find
    a handful of actual bugs in them. This simple heuristic filter
    saves you the trouble by discarding crashes that seem to be similar
    to another one you've already found.

### Function: __add_crash(self, crash)

### Function: __add_memory(self, crash_id, memoryMap)

### Function: find(self, signature, order, since, until, offset, limit)

**Description:** Retrieve all crash dumps in the database, optionally filtering them by
signature and timestamp, and/or sorting them by timestamp.

Results can be paged to avoid consuming too much memory if the database
is large.

@see: L{find_by_example}

@type  signature: object
@param signature: (Optional) Return only through crashes matching
    this signature. See L{Crash.signature} for more details.

@type  order: int
@param order: (Optional) Sort by timestamp.
    If C{== 0}, results are not sorted.
    If C{> 0}, results are sorted from older to newer.
    If C{< 0}, results are sorted from newer to older.

@type  since: datetime
@param since: (Optional) Return only the crashes after and
    including this date and time.

@type  until: datetime
@param until: (Optional) Return only the crashes before this date
    and time, not including it.

@type  offset: int
@param offset: (Optional) Skip the first I{offset} results.

@type  limit: int
@param limit: (Optional) Return at most I{limit} results.

@rtype:  list(L{Crash})
@return: List of Crash objects.

### Function: find_by_example(self, crash, offset, limit)

**Description:** Find all crash dumps that have common properties with the crash dump
provided.

Results can be paged to avoid consuming too much memory if the database
is large.

@see: L{find}

@type  crash: L{Crash}
@param crash: Crash object to compare with. Fields set to C{None} are
    ignored, all other fields but the signature are used in the
    comparison.

    To search for signature instead use the L{find} method.

@type  offset: int
@param offset: (Optional) Skip the first I{offset} results.

@type  limit: int
@param limit: (Optional) Return at most I{limit} results.

@rtype:  list(L{Crash})
@return: List of similar crash dumps found.

### Function: count(self, signature)

**Description:** Counts how many crash dumps have been stored in this database.
Optionally filters the count by heuristic signature.

@type  signature: object
@param signature: (Optional) Count only the crashes that match
    this signature. See L{Crash.signature} for more details.

@rtype:  int
@return: Count of crash dumps stored in this database.

### Function: delete(self, crash)

**Description:** Remove the given crash dump from the database.

@type  crash: L{Crash}
@param crash: Crash dump to remove.

### Function: decorator(w)

**Description:** The C{decorator} module was not found. You can install it from:
U{http://pypi.python.org/pypi/decorator/}

### Function: d(fn)

### Function: x()
