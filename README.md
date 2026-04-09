# Writing Assist

Phase 0 scaffold for a local-first writing workspace.

## Stack

- `Tauri`
- `SvelteKit`
- `CodeMirror`
- `Rust`
- `SQLite`
- `Rig`
- `docker-compose`

## Development

This machine uses legacy `docker-compose`, not `docker compose`.

### Build the workspace image

```sh
docker-compose build workspace
```

### Install dependencies inside the container

```sh
docker-compose run --rm workspace ./scripts/bootstrap.sh
```

### Start an interactive shell

```sh
docker-compose run --rm --service-ports workspace bash
```

### Run the frontend dev server from inside the container

```sh
npm run dev
```

Default forwarded ports:

- dev server: `5180`
- preview server: `4180`

### Run Rust metadata checks from inside the container

```sh
cargo metadata --format-version 1 --no-deps
```

## Notes

- The project folder is mounted into `/workspace`.
- Node modules and Cargo caches are persisted through named Docker volumes.
- The Tauri desktop entrypoint is scaffolded in `src-tauri/`, but GUI runtime concerns are deferred until later phases.
