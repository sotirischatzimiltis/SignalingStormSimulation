FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No ENTRYPOINT/CMD baked in -- mcp_server and the 3 agents are all the same
# codebase/image, differentiated only by docker-compose.yml's per-service
# `command:` override (python -m mcp_server / python -m agents.tactical / ...).
