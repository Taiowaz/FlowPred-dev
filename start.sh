#!/bin/bash

# Start gunicorn with the correct parameters
exec gunicorn \
    --bind 0.0.0.0:5005 \
    app:app 