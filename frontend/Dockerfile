FROM node:13.10 as base

WORKDIR /app

FROM base as builder

# Install dependencies
COPY yarn.lock package.json ./
RUN yarn --version
RUN yarn install --frozen-lockfile

COPY . /app

FROM base as final

# Copy app and dependencies from builder
COPY --from=builder /app /app