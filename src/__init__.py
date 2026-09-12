"""Initialize package with system certificate support on Windows."""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
