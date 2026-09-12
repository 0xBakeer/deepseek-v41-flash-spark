CREATE TABLE IF NOT EXISTS services (
    id            BIGSERIAL PRIMARY KEY,
    name          TEXT        NOT NULL UNIQUE,
    owner         TEXT        NOT NULL DEFAULT 'unassigned',
    state         TEXT        NOT NULL CHECK (state IN ('healthy', 'degraded', 'down')),
    p99_ms        INTEGER     CHECK (p99_ms >= 0),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX services_state_idx ON services (state) WHERE state <> 'healthy';
CREATE INDEX services_owner_idx ON services (owner, name);

CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER services_touch BEFORE UPDATE ON services
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

WITH recent AS (
    SELECT owner, count(*) AS n, avg(p99_ms)::numeric(10,1) AS avg_p99
    FROM services
    WHERE updated_at > now() - interval '7 days'
    GROUP BY owner
)
SELECT r.owner, r.n, r.avg_p99,
       rank() OVER (ORDER BY r.avg_p99 DESC NULLS LAST) AS slowest_rank
FROM recent r
WHERE r.n >= 3
ORDER BY slowest_rank
LIMIT 20;
