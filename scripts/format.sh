#!/bin/bash

isort ../src
black --line-length 79 ../src

