#! /bin/bash
docker compose down
docker compose up --build --force-recreate --remove-orphans
