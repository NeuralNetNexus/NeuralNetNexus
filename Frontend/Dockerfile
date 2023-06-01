FROM node:latest
WORKDIR /usr/app

COPY package.json ./
RUN npm install

COPY . .

ENTRYPOINT [ "node", "index.js" ]