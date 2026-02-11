# PROD 2026 - Look-a-Like

## Important

Russian version: [README.md](README.md)

Task description version: 1

Use this repository as a solution structure template.

## What is considered your submission

The evaluated submission is the latest commit in the `main` branch.

## Minimum required repository structure

The repository must include:

1. `Dockerfile`
2. `docker-compose.yml`
3. `dvc.yaml`
4. `params.yaml`

These files are validated by the sample `.gitlab-ci.yml` in this template.

## Runtime expectations

During evaluation, after `docker compose up -d`, your service must:

1. listen on port `80`;
2. return `200` on `GET /ready` within 180 seconds;
3. implement endpoints according to `openapi.yml`.

## Academic integrity

If you use third-party code, provide attribution. Submissions are checked for plagiarism.
