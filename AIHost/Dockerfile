# См. статью по ссылке https://aka.ms/customizecontainer, чтобы узнать как настроить контейнер отладки и как Visual Studio использует этот Dockerfile для создания образов для ускорения отладки.

# Этот этап используется при запуске из VS в быстром режиме (по умолчанию для конфигурации отладки)
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS base
USER root
WORKDIR /app
EXPOSE 8080
EXPOSE 8081


# Этот этап используется для сборки проекта службы
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["AIHost/AIHost.csproj", "AIHost/"]
RUN dotnet restore "./AIHost/AIHost.csproj"
COPY . .
WORKDIR "/src/AIHost"
RUN dotnet build "./AIHost.csproj" -c $BUILD_CONFIGURATION -o /app/build

# Этот этап используется для публикации проекта службы, который будет скопирован на последний этап
FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./AIHost.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false


FROM base AS pre_final
# 1. Расширяем список репозиториев (добавляем contrib, non-free и non-free-firmware)
# Команда хитрая, так как покрывает и старый формат Debian, и новый (DEB822)
RUN sed -i 's/Components: main/Components: main contrib non-free non-free-firmware/g' /etc/apt/sources.list.d/debian.sources || true \
    && sed -i 's/main/main contrib non-free non-free-firmware/g' /etc/apt/sources.list || true

# 2. Устанавливаем наш "швейцарский нож"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    mesa-opencl-icd \
    # Немного обновили имя пакета (в новых Debian libgl1 надежнее старого libgl1-mesa-glx)
    libgl1 \
    libglib2.0-0 \
    nvidia-cuda-toolkit \
    pciutils \
    lshw \
    libamdhip64-dev \
    rocm-smi \
    && rm -rf /var/lib/apt/lists/*

# Этот этап используется в рабочей среде или при запуске из VS в обычном режиме (по умолчанию, когда конфигурация отладки не используется)
FROM pre_final AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "AIHost.dll"]