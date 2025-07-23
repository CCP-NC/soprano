# Test Save Directory

This directory is used by Soprano's test suite to write temporary files during test execution.

Please do not delete this directory or the `.gitkeep` file within it, as they are necessary for tests to run correctly.

## For developers

When running tests, this directory needs to have write permissions. If you encounter file permission issues in tests, run:

```bash
chmod -R 777 tests/test_save
```

## GitHub Actions and Act

The CI workflow ensures this directory has proper permissions. When running tests locally with Act, use the provided helper script:

```bash
./run_act_tests.sh
```
