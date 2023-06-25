#!/bin/sh
# Replace the placeholders with the actual SERVER_URI and SOCKET_URI values 
find /usr/share/nginx/html -type f -print0 | xargs -0 sed -i "s|__SERVER_URI__|${SERVER_URI:-localhost}|g"
find /usr/share/nginx/html -type f -print0 | xargs -0 sed -i "s|__SOCKET_URI__|${SOCKET_URI:-localhost}|g"
