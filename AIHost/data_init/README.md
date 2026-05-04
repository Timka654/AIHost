# Data Initialization Files

This folder contains template files that are copied to the `data/` directory when the server starts for the first time or when the data directory is empty.

## Files:

- **model.schema.json** - JSON schema for model configurations
- **model.example.json** - Example model configuration file
- **config/server.config.json** - Default server configuration

## Purpose:

When AIHost starts, it checks if the `data/` directory exists and contains the necessary files. If not, it automatically copies these templates to set up the initial configuration.

## Customization:

You can modify these files to change the default initial configuration for new AIHost installations.
