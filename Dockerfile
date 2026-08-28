# gridcast — container image for the forecast API.
#
# Two stages. The builder has a compiler and pip's build machinery; the runtime
# has neither, and receives only the finished virtual environment. That is
# worth doing for more than image size: a runtime with no compiler and no
# package manager is a runtime where a process that gets compromised has much
# less to work with.
#
# What goes in is the `serve` extra, not `analysis`. The service loads models
# and answers requests — it never draws a chart, so matplotlib and its font and
# image libraries have no business being here. Charts are rendered by
# `gridcast report` on a developer machine.
#
# What does NOT go in: the database and the trained models. Both are artifacts,
# not source. Baking a database into an image gives every container a private
# copy of stale data and makes the image grow every time the data does; they
# are mounted at runtime instead. See docker-compose.yml.

# ---------------------------------------------------------------- builder
FROM python:3.12-slim AS builder

# Pip writes noisy version warnings and caches wheels it will never reuse in a
# layer that is about to be discarded.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy only what the install needs, so editing a source file does not
# invalidate the dependency layer. This is the difference between a rebuild
# that takes two seconds and one that takes two minutes.
COPY pyproject.toml README.md ./
COPY gridcast/__init__.py gridcast/

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install ".[serve]"

# Now the real source, and install the package itself over the top.
COPY gridcast/ gridcast/
RUN /opt/venv/bin/pip install --no-deps .

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

# PYTHONUNBUFFERED matters more in a container than anywhere else. Without it
# Python buffers stdout when it is not a terminal, so `docker logs` shows
# nothing until the buffer fills — and a service that appears to log nothing
# looks like a service that has hung.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# A non-root user, created before the volumes are made so it owns them. Running
# as root inside a container is the default and is worth undoing: it costs one
# line and removes a whole class of consequence from any escape.
RUN useradd --create-home --uid 10001 gridcast \
 && mkdir -p /data /models \
 && chown -R gridcast:gridcast /data /models

COPY --from=builder /opt/venv /opt/venv

USER gridcast
WORKDIR /home/gridcast

VOLUME ["/data", "/models"]
EXPOSE 8000

# Uses Python rather than curl, because curl is not in the slim image and
# installing a whole HTTP client to answer one question would undo the point of
# using slim. `/health` returns 200 while the service is alive and reports
# "degraded" in the body if no models loaded — so this checks liveness, which
# is what a healthcheck is for, and leaves readiness to the caller.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# --host 0.0.0.0 is not optional here, and the reason catches people out.
# The CLI defaults to 127.0.0.1, which is correct on a laptop and useless in a
# container: it binds the container's own loopback, so the published port
# connects to nothing and the service looks broken while running perfectly.
ENTRYPOINT ["python", "-m", "gridcast"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--db", "/data/gridcast.db", "--model-dir", "/models"]
