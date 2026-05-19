from __future__ import annotations
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from nnm.config import get_settings
from nnm.db.base import Base
import nnm.db.models  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def include_object(obj, name, type_, reflected, compare_to):
    if type_ == "table":
        info = getattr(obj, "info", {}) or {}
        if info.get("managed_by") == "operational":
            return False
    return True


target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=get_settings().db_url, target_metadata=target_metadata,
        literal_binds=True, dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(
        connection=connection, target_metadata=target_metadata,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = get_settings().db_url
    connectable = async_engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
