-- Create the dedicated Phoenix database on first boot.
--
-- This file is loaded by the official postgres entrypoint via psql
-- against the maintenance DB (POSTGRES_DB). CREATE DATABASE cannot
-- run inside a transaction block, so this file must be the only
-- top-level statement and must not be combined with other DDL.
\set ON_ERROR_STOP on
CREATE DATABASE phoenix;
