-- Enable the pgvector extension in the application database.
--
-- Runs after the application database is created. The phoenix database
-- is left untouched; it does not need pgvector.
\connect groundgraph
CREATE EXTENSION IF NOT EXISTS vector;
