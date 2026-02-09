from semantic_id.uniqueness.resolver import UniqueIdResolver
from semantic_id.uniqueness.stores import InMemoryCollisionStore, SQLiteCollisionStore


def test_in_memory_resolver():
    resolver = UniqueIdResolver(store=InMemoryCollisionStore())

    ids = ["a", "b", "a", "c", "a", "b"]
    unique = resolver.assign(ids)

    expected = ["a", "b", "a-1", "c", "a-2", "b-1"]
    assert unique == expected


def test_sqlite_resolver(tmp_path):
    db_path = str(tmp_path / "collisions.db")
    store = SQLiteCollisionStore(db_path=db_path)
    resolver = UniqueIdResolver(store=store)

    ids = ["x", "x", "y", "x"]
    unique = resolver.assign(ids)

    expected = ["x", "x-1", "y", "x-2"]
    assert unique == expected

    # Test persistence (re-using same DB)
    store2 = SQLiteCollisionStore(db_path=db_path)
    resolver2 = UniqueIdResolver(store=store2)

    unique2 = resolver2.assign(["x"])
    assert unique2 == ["x-3"]
