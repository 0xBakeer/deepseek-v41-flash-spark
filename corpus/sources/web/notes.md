# Incident review: dashboard latency, 2026-08-14

## Summary

Between 09:12 and 10:48 UTC the service dashboard served p99 latencies above four seconds, against a
normal p99 of 180 ms. No requests were dropped and no data was lost. The cause was a missing index
after a schema migration, compounded by a connection pool that was sized for the old query plan.

## Timeline

- **09:12** A migration adds the `state` column and backfills it for 2.1 million rows.
- **09:14** The first alert fires on p99 above one second. On-call acknowledges at 09:19.
- **09:31** Read replicas show sequential scans on `services` for every dashboard query.
- **09:48** The pool saturates; requests queue rather than fail, so error rate stays flat and the
  usual error-rate alert never fires.
- **10:20** A partial index on `state` is created concurrently.
- **10:48** p99 returns to baseline.

## What went wrong

The migration was reviewed for correctness but not for query plans. Our review checklist asks
whether a migration is reversible and whether it locks, and it does not ask what the planner will do
afterwards. The second failure is the alert: latency was the only signal, and it fired late because
the pool absorbed the pressure before anything visibly broke.

## Actions

1. Add a plan check to the migration template: every migration that adds a filterable column ships
   with the index it needs, or an explicit note saying why it does not.
2. Alert on pool saturation directly, not only on the latency it eventually causes.
3. Backfill in batches so a long transaction cannot hold a snapshot open across a deploy.
