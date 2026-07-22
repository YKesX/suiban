# Vendored SLAP schemas: provenance

These 10 JSON Schema files are a **vendored, byte-identical copy** of the frozen SLAP
1.0 schema set. The **canonical source is the separate `slap` repo** (`slap/schemas/`);
this directory is suiban's independent, self-contained copy.

suiban does not import the `slap` package. It vendors these schemas and implements the
protocol itself, exactly as it implements `docs/api.md` independently of its HTTP
clients. The one coordination point is the frozen shape, not shared code.

Do not hand-edit these files. They must stay byte-identical to the canonical `slap`
repo; `tests/test_slap.py::test_vendored_schemas_are_byte_identical_to_slap_repo` is the
drift guard (it skips when the sibling `slap` repo is not checked out alongside suiban).
To adopt a protocol change, re-copy from the canonical repo after it version-bumps.
