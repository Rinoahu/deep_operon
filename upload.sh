#!/bin/bash

git config --global user.email xiaohu@iastate.edu
git config --global user.name Rinoahu


git remote rm origin

git add -A .
git commit -m 'change the project name'
git remote add origin https://github.com/Rinoahu/deep_operon

git pull origin master
git push origin master

git checkout master
