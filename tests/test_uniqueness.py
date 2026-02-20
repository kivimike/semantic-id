from semantic_id.uniqueness.resolver import UniqueIdResolver
from semantic_id.uniqueness.stores import InMemoryCollisionStore, SQLiteCollisionStore


def test_in_memory_resolver():
    resolver = UniqueIdResolver(store=InMemoryCollisionStore())

    ids = ["a", "b", "a", "c", "a", "b"]
    unique = resolver.assign(ids)

    expected = ["a", "b", "a-1", "c", "a-2", "b-1"]
    assert unique == expected


def test_empty_input():
    resolver = UniqueIdResolver(store=InMemoryCollisionStore())
    assert resolver.assign([]) == []


def test_single_item():
    resolver = UniqueIdResolver(store=InMemoryCollisionStore())
    assert resolver.assign(["x"]) == ["x"]


def test_all_unique():
    resolver = UniqueIdResolver(store=InMemoryCollisionStore())
    ids = ["a", "b", "c", "d"]
    assert resolver.assign(ids) == ids


def test_all_collisions():
    resolver = UniqueIdResolver(store=InMemoryCollisionStore())
    ids = ["x", "x", "x", "x"]
    expected = ["x", "x-1", "x-2", "x-3"]
    assert resolver.assign(ids) == expected


def test_item_ids_on_collision():
    resolver = UniqueIdResolver(store=InMemoryCollisionStore())
    sids = ["3-9-1", "3-9-1", "5-0-2"]
    item_ids = ["SKU100", "SKU200", "SKU300"]
    unique = resolver.assign(sids, item_ids=item_ids)
    assert unique == ["3-9-1", "3-9-1-SKU200", "5-0-2"]


def test_item_ids_no_collision():
    resolver = UniqueIdResolver(store=InMemoryCollisionStore())
    sids = ["a", "b", "c"]
    item_ids = ["id1", "id2", "id3"]
    unique = resolver.assign(sids, item_ids=item_ids)
    assert unique == ["a", "b", "c"]


def test_custom_sep_on_suffix():
    resolver = UniqueIdResolver(store=InMemoryCollisionStore())
    sids = ["1/2/3", "1/2/3"]
    unique = resolver.assign(sids, sep="/")
    assert unique == ["1/2/3", "1/2/3/1"]


def test_item_ids_with_custom_sep():
    resolver = UniqueIdResolver(store=InMemoryCollisionStore())
    sids = ["a", "a"]
    unique = resolver.assign(sids, item_ids=["x1", "x2"], sep="::")
    assert unique == ["a", "a::x2"]


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
