import sqlite3
import json

db = sqlite3.connect("data/vector_store/chroma.sqlite3")

config = {
    "hnsw_configuration": {
        "space": "cosine",
        "ef_construction": 100,
        "ef_search": 10,
        "num_threads": 16,
        "M": 16,
        "resize_factor": 1.2,
        "batch_size": 100,
        "sync_threshold": 1000,
        "_type": "HNSWConfigurationInternal"
    },
    "_type": "CollectionConfigurationInternal"
}

db.execute(
    "UPDATE collections SET config_json_str = ? WHERE name = ?",
    (json.dumps(config), "study_notes")
)

db.commit()

print("Updated rows:", db.total_changes)

row = db.execute(
    "SELECT name, config_json_str FROM collections WHERE name = ?",
    ("study_notes",)
).fetchone()

print(row)

db.close()
