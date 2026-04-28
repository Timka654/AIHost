# Syntax reference: https://docs.docker.com/engine/reference/builder/
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy project files
COPY ["AIHost.csproj", "./"]
RUN dotnet restore "AIHost.csproj"

# Copy source code
COPY . .

# Build
RUN dotnet build "AIHost.csproj" -c Release -o /app/build

# Publish
RUN dotnet publish "AIHost.csproj" -c Release -o /app/publish /p:UseAppHost=false

# Runtime image
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app

# Install Vulkan drivers (for GPU support)
RUN apt-get update && \
    apt-get install -y vulkan-tools libvulkan1 && \
    rm -rf /var/lib/apt/lists/*

# Copy published app
COPY --from=build /app/publish .

# Create data directories
RUN mkdir -p /app/data/config /app/data/models /app/data/logs /app/data/cache

# Volume for persistent data
VOLUME ["/app/data"]

# Expose port
EXPOSE 11434

# Set environment variables
ENV ASPNETCORE_URLS=http://+:11434
ENV DOTNET_RUNNING_IN_CONTAINER=true

# Entry point
ENTRYPOINT ["dotnet", "AIHost.dll", "--web"]
