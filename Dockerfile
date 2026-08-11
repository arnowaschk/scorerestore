FROM python:3.12.11-slim-bookworm AS lilypond

ARG LILYPOND_VERSION=2.26.0
ARG LILYPOND_SHA256=cd8a097a9f52cb2b9f4e7914774786f203f4fc61fcd299afcbb63c23fa5c6b24

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl \
    && curl --fail --location --silent --show-error \
       "https://gitlab.com/lilypond/lilypond/-/releases/v${LILYPOND_VERSION}/downloads/lilypond-${LILYPOND_VERSION}-linux-x86_64.tar.gz" \
       --output /tmp/lilypond.tar.gz \
    && echo "${LILYPOND_SHA256}  /tmp/lilypond.tar.gz" | sha256sum --check --strict \
    && mkdir -p /opt \
    && tar -xzf /tmp/lilypond.tar.gz -C /opt \
    && rm /tmp/lilypond.tar.gz

FROM ghcr.io/astral-sh/uv:0.11.20 AS uv

FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

COPY --from=uv /uv /uvx /bin/
COPY --from=lilypond /opt/lilypond-2.26.0 /opt/lilypond-2.26.0

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY assets ./assets
COPY configs ./configs
COPY src ./src

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:/opt/lilypond-2.26.0/bin:${PATH}"

RUN mkdir -p /data /models /runs

CMD ["scorerestore", "--help"]
