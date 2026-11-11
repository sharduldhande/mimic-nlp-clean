import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from collections.abc import Iterator
import pandas as pd

import spacy; import medspacy
import pyarrow.parquet as pq

from typing import Union, Iterable, Optional, Dict, Any, Tuple, List, Literal, Set

nlp = medspacy.load(medspacy_enable=["medspacy_sectionizer"])


table = pq.read_table(
    "data_cleaned/dataset_splits_000000000000.parquet",
    columns = ["note_id", "discharge_note"]
).slice(0,1)

note_id = table.column("note_id")[0].as_py()
text = table.column("discharge_note")[0].as_py()

doc = nlp(text)
print("note:", note_id)
print(doc._.section_categories)


#https://stackoverflow.com/questions/50755586/how-to-loop-large-parquet-file-with-generators-in-python



def iter_by_line_parquet(path: str, batch_size: int) -> Iterator[dict[str, Any]]:
    """Iterate over a parquet file line by line.

    Each line is represented by a dict.

    Args:
        path: path to the .parquet file.
        batch_size: number of rows to load in memory.

    Yields:
        line as dict.
    """
    parquet_file = pq.ParquetFile(path)

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["note_id", "discharge_note"]):  # changed
        yield from batch.to_pylist()


# path = "data_cleaned/dataset_splits_000000000000.parquet"  # changed
# batch_size = 256  # changed
#
# # print the first row of your parquet file
# print(next(iter_by_line_parquet(path, batch_size)))
#
# rows = []
#
# for row in iter_by_line_parquet(path, batch_size):
#     doc = nlp(row["discharge_note"])
#     for cat, body in zip(doc._.section_categories, doc._.section_bodies):
#         rows.append({"note_id": row["note_id"], "section": cat, "n_chars": len(body.text)})
#
# pd.DataFrame(rows).to_parquet("medspacy_sections1.parquet", index=False)
# print("saved", len(rows), "rows")

path = "data_cleaned/dataset_splits_000000000001.parquet"  # changed
batch_size = 256  # changed

# print the first row of your parquet file
print(next(iter_by_line_parquet(path, batch_size)))

rows = []

for row in iter_by_line_parquet(path, batch_size):
    doc = nlp(row["discharge_note"])
    for cat, body in zip(doc._.section_categories, doc._.section_bodies):
        rows.append({"note_id": row["note_id"], "section": cat, "n_chars": len(body.text)})

pd.DataFrame(rows).to_parquet("medspacy_sections2.parquet", index=False)
print("saved", len(rows), "rows")