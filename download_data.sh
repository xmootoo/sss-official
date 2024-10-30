#!/bin/bash

# Check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown is not installed. Installing now..."
    pip install gdown
fi

# Check if unzip is installed
if ! command -v unzip &> /dev/null
then
    echo "unzip is not installed. Please install it using your package manager."
    exit 1
fi

# File ID from the Google Drive link
FILE_ID="1x_wEUaJ83ljaCUxUeIwhrUqXa-Qd6pSx"
ZIP_FILE="data.zip"

# Download the file
echo "Downloading file..."
gdown https://drive.google.com/uc?id=$FILE_ID -O $ZIP_FILE

# Check if download was successful
if [ ! -f "$ZIP_FILE" ]; then
    echo "Download failed. Exiting script."
    exit 1
fi

echo "Download complete!"

# Unzip the file
echo "Unzipping $ZIP_FILE..."
unzip -q $ZIP_FILE

# Check if unzip was successful
if [ $? -ne 0 ]; then
    echo "Unzip failed. Keeping the zip file for troubleshooting."
    exit 1
fi

echo "Unzip complete!"

# Delete the original zip file
echo "Deleting $ZIP_FILE..."
rm $ZIP_FILE

echo "Process completed successfully!"
