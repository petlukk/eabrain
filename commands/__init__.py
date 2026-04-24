"""eabrain CLI command handlers.

Split by concern:
  - memory: observation and session CRUD, sync
  - search: kernel/observation search, ref lookup, pattern browsing
  - system: status, inject, init, serve

Handlers import helpers from `eabrain` at call time (function-local imports)
to avoid a module-load cycle. eabrain.py imports from commands.* at the top of
`main()`, which is also call time — so neither module references the other at
module load.
"""
