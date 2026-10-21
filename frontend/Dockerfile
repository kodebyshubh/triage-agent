# Frontend: build the Vite app, then serve the static files with an unprivileged
# nginx (runs as a non-root user and listens on 8080).
FROM node:22-alpine AS build
WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Vite inlines VITE_* variables at build time.
ARG VITE_API_URL=http://localhost:8000
ARG VITE_API_KEY=
ENV VITE_API_URL=$VITE_API_URL \
    VITE_API_KEY=$VITE_API_KEY
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -qO /dev/null http://127.0.0.1:8080/ || exit 1
