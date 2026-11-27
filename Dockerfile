# A container for `orbix serve`.
#
# The binary is statically linked with CGO_ENABLED=0 -- verified: `ldd` reports "not a dynamic
# executable" and the ELF has no dynamic section -- so the runtime image needs no libc, no
# libgcc, and no shell. `scratch` is therefore the honest base: it holds the binary, the CA
# bundle for outbound OTLP, and nothing else.
#
# What this image is NOT: it does not include a model. A GGUF checkpoint is hundreds of
# megabytes to several gigabytes and belongs in a volume or a derived image, not here. Mount one
# at /models and point --model at it, or build on top of this stage.
#
#   docker build -t orbix .
#   docker run --rm -p 8080:8080 -v /path/to/models:/models:ro \
#     orbix --model /models/your-model.gguf

# ---------------------------------------------------------------- build
FROM --platform=$BUILDPLATFORM golang:1.27 AS build

# The target platform, so one Dockerfile builds a correct image for each architecture a
# manifest lists. Without it the build stage cross-compiles with the builder's GOARCH -- on an
# amd64 runner every arm64 image would carry an amd64 binary, and the failure would only appear
# when someone ran it.
#
# --platform=$BUILDPLATFORM on FROM is the other half: build on the runner's own architecture
# and cross-compile, rather than running the whole Go toolchain under emulation.
ARG TARGETOS
ARG TARGETARCH

WORKDIR /src

# The module has no dependencies outside the standard library, so there is no go.sum to copy and
# no `go mod download` step to cache. Copy the sources and build.
COPY . .

# The same flags `just build` uses, minus the version metadata: a container build has no
# guarantee that .git is present, and stamping a wrong commit hash is worse than stamping none.
# Pass the release metadata explicitly when building a tagged image:
#   --build-arg VERSION=v0.1.0 --build-arg COMMIT=abc1234
ARG VERSION=
ARG COMMIT=
ARG DATE=
# -buildvcs=false because .dockerignore excludes .git: without it the go command fails with
# "error obtaining VCS status: exit status 128" and the build stops. Verified by building from
# the exact filtered context. The version metadata that -buildvcs would have derived comes from
# the --build-arg values below instead.
# TARGETARCH is empty under a plain `docker build` -- only buildx sets it -- and the go command
# treats `GOARCH=` as unset, so such a build uses the builder's own architecture, which is what
# a single-arch image should be. The two cases need no separate branch.
RUN CGO_ENABLED=0 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH} go build \
        -buildvcs=false \
        -trimpath \
        -ldflags "-s -w \
            -X main.version=${VERSION:-dev} \
            -X main.commit=${COMMIT:-none} \
            -X main.date=${DATE:-unknown}" \
        -o /out/orbix ./cmd/orbix

# ---------------------------------------------------------------- runtime
FROM scratch

# Trace export is OTLP over HTTP, which means TLS to any real collector. `scratch` has no
# certificate store, so without this the exporter fails on every span with "x509: certificate
# signed by unknown authority" and increments trace_spans_dropped_total -- a failure that is
# visible only in /metrics, which is why the bundle is copied rather than left out.
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt

COPY --from=build /out/orbix /orbix

# The default address is ":8080", which is every interface. Inside a container that is what a
# port mapping needs, so it is left alone -- but note that the server has no authentication
# (docs/usage.md, "No CORS and no authentication"), so publish it only on a network you trust or
# put an authenticating proxy in front.
# A container process runs as root unless told otherwise, and root in a container is root on
# the host's kernel. The binary needs no privileges: it reads a checkpoint, holds a port above
# 1024, and writes nothing. A numeric uid is used because `scratch` has no /etc/passwd -- a name
# would resolve to nothing and `docker run --user nobody` would silently fall back to root's
# filesystem view while still reporting a name.
#
# 65534 is nobody:nogroup, conventional for "a process that needs no identity". Exported so a
# volume can be given matching ownership without guessing the number.
ARG UID=65534
ARG GID=65534
USER $UID:$GID

EXPOSE 8080

# `orbix healthcheck` is the probe, not a shell one-liner: `scratch` has no curl, no wget and no
# shell, so the http client has to be the binary. It asks for /healthz and decodes the body, so a
# 200 from some unrelated process on the same port does not pass.
#
# /healthz is answered from the HTTP layer after the engine exists (serve builds the engine, then
# calls net.Listen), which is what makes a 200 mean "weights resident" rather than "process not
# dead". --start-period covers a load that takes longer than the interval: measured, Qwen2.5-0.5B
# at 1.84 GiB takes about 2.9 s, and a paged 7B checkpoint on a slow disk can take a minute.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["/orbix", "healthcheck", "--addr", "http://127.0.0.1:8080", "--quiet"]

ENTRYPOINT ["/orbix"]
CMD ["serve", "--model", "/models/model.gguf", "--addr", ":8080"]
