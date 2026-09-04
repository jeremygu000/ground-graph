-- Initial database setup for graphrag
-- This file is executed by the official postgres entrypoint on first boot.

-- Ensure pgvector extension can be created.
-- pgvector extension is provided by the pgvector/pgvector image; we still
-- try CREATE EXTENSION in case the image is swapped out for plain postgres.
CREATE EXTENSION IF NOT EXISTS vector;
