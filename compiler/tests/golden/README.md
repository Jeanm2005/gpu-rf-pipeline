# Golden tests

Each `<name>.sf` here is one case. `tests/CMakeLists.txt` registers every one of them
automatically (`add_sigflowc_golden_tests()` in `cmake/SigflowGoldenTest.cmake`): CTest runs

```bash
sigflowc <flags> <name>.sf      # from this directory, so diagnostics print the bare file name
```

and compares its stdout and stderr together, byte for byte, against `<name>.expected`. A
non-zero exit code is recorded as a final `[exit N]` line, so negative cases pin both the
diagnostics and the failure.

The flags come from the first line of the program:

```
# args: --emit=tokens
```

Naming: `lex_*`, `parse_*`, `sema_*`, ... for the stage a case exercises; positive and negative
cases sit side by side (every grammar rule has at least one of each).

To add or refresh golden files:

```bash
ctest --test-dir build -R golden_<name> --output-on-failure   # see what changed
./build/sigflowc --update-golden tests/golden                  # regenerate every .expected
git diff tests/golden                                          # review before committing
```

`cmake --build build --target update_golden_<name>` regenerates a single case through the same
CMake runner CTest uses. Every golden change is explained in the commit message that makes it.
