#!/usr/bin/env bash

# update firefox to prevent crashes
sudo apt-get update
sudo apt-get install --only-upgrade firefox
sudo apt-get install tree

# configure git repo locally
git config user.name "Matthias Budde"
git config user.email "buddemat@users.noreply.github.com"

