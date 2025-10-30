# Use official Node.js LTS image
FROM node:20-slim

# Install OS dependencies for Expo/React Native
RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json first (for caching npm install)
COPY package.json package-lock.json* ./

# Install project dependencies
RUN npm install --legacy-peer-deps

# Copy the rest of the app
COPY . .

# Expose Expo default port
EXPOSE 19000
EXPOSE 19001
EXPOSE 19002

# Default command
ENTRYPOINT ["sh", "-c"]
CMD ["npm run generate && npm run start"]
