# Stage 1: build
FROM node:20-alpine AS builder
WORKDIR /app

# Install deps
COPY package.json package-lock.json* ./
RUN npm ci --silent || npm install --silent

# Build
COPY . .
# Default to relative paths so the nginx sidecar in the dashboard
# container can forward /api and /ws to the backend. Override at
# build time with --build-arg VITE_API_URL=... for non-proxy setups.
ARG VITE_API_URL=
ARG VITE_WS_URL=
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_WS_URL=$VITE_WS_URL
RUN npm run build

# Stage 2: serve
FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -qO- http://localhost/ > /dev/null || exit 1
CMD ["nginx", "-g", "daemon off;"]
