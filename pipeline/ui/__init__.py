"""`readaloud ui`: the command line's operations in a browser.

A small HTTP server from the standard library (as services/kokoro is) and a page of plain JavaScript; nothing to
install beyond the pipeline itself. It listens on this machine only. Everything that changes a book goes through
the same code as the command line: long jobs are `readaloud run` processes that outlive the browser tab, and the
files a person may edit (sections.yaml, genders.yaml, attribution_overrides.csv, the cast file, config.yaml) stay
the files they are.
"""
