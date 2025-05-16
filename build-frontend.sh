docker run --rm -v ./frontend:/home/node/frontend -w /home/node/frontend docker.io/node:22-alpine sh -c "npm install && npm run build"
