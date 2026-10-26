# Golden tests

Each case is registered in `tests/CMakeLists.txt` with `add_golden_test()`
(`cmake/SigflowGoldenTest.cmake`): it runs a command and compares its stdout, byte for byte,
against a committed `<name>.expected` file.

There are no `sigflowc` cases yet: the binary doesn't exist until the lexer, parser, and
semantic analysis are written (see `../../docs/ROADMAP.md`, phase A). `harness_smoke` is a
placeholder that runs `cmake -E echo` instead, to prove the mechanism works; delete it once the
first real case (for example an empty-program or lexer golden) is registered.

Once `sigflowc` exists, a typical case will look like:

```cmake
add_golden_test(empty_program
  COMMAND $<TARGET_FILE:sigflowc> ${CMAKE_CURRENT_SOURCE_DIR}/golden/empty_program.sf --emit=tokens
  EXPECTED ${CMAKE_CURRENT_SOURCE_DIR}/golden/empty_program.expected
)
```

To add or refresh a golden file:

```bash
ctest --test-dir build -R golden_<name> --output-on-failure   # see what changed
cmake --build build --target update_golden_<name>              # regenerate EXPECTED
git diff tests/golden/<name>.expected                          # review before committing
```

Per `compiler/AGENTS.md`, a golden change must be explained in the commit message, and this is
done only when the owner asks.
