FROM node:20-alpine

WORKDIR /usr/app

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 3003

CMD [ "node", "index.js" ]