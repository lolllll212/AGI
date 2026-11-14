Put the certificate chain and private key here as `fullchain.pem` and `privkey.pem`.
Git ignores everything in this folder except this file.

For a local trial, a self-signed pair is enough:

    openssl req -x509 -newkey rsa:2048 -nodes -days 30 -subj "/CN=localhost" \
      -keyout monitoring/nginx/ssl/privkey.pem -out monitoring/nginx/ssl/fullchain.pem
