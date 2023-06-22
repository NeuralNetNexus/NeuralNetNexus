# Stage 1: Build the React app
FROM node:16-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install --silent
COPY . ./
RUN npm run build

# Stage 2: Serve the app using Nginx 
FROM nginx:latest
COPY --from=build /app/build /usr/share/nginx/html
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx/nginx.conf /etc/nginx/conf.d

RUN chmod -R 755 /usr/share/nginx/html

COPY ./cmd.sh /docker-entrypoint.d/

RUN chmod +x /docker-entrypoint.d/cmd.sh

# EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]