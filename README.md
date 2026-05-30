## 1. System Requirements
- A computer with **Docker** and **Docker Compose** installed (Docker Desktop for Windows environments).
- Ensure the Docker service is running before executing commands.

## 2. Setting Up Model Weights
To optimize Git repository storage, model weight files (`.pt`, `.pth`) have been excluded. You must manually download and provide the weights before running the system:

1. Access the storage link to download the weights: `[https://drive.google.com/file/d/1vO_dCxF5NPyAusCvpWLgsj3sRG6taoa7/view?usp=sharing]`
2. Download the required weight files to your computer.
3. Copy and place these files into the `models/` directory located at the root of the project. 
*(Example: `models/yolov10/best.pt` or `models/faster-rcnn/model.pth`)*.
(Note: The Docker system is configured to automatically map this directory into the Backend container).

## 3. Starting the System
Open a command-line interface (Terminal/PowerShell) at the root directory of the project (where the `docker-compose.yml` file is located) and execute the following command:

```bash
docker-compose up --remove-orphans