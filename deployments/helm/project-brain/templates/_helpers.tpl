{{/*
Expand the name of the chart.
*/}}
{{- define "project-brain.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "project-brain.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart label
*/}}
{{- define "project-brain.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "project-brain.labels" -}}
helm.sh/chart: {{ include "project-brain.chart" . }}
{{ include "project-brain.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "project-brain.selectorLabels" -}}
app.kubernetes.io/name: {{ include "project-brain.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account
*/}}
{{- define "project-brain.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "project-brain.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
PostgreSQL hostname
*/}}
{{- define "project-brain.postgresql.host" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "project-brain.fullname" .) }}
{{- else }}
{{- .Values.externalPostgresql.host }}
{{- end }}
{{- end }}

{{/*
Neo4j hostname
*/}}
{{- define "project-brain.neo4j.host" -}}
{{- if .Values.neo4j.enabled }}
{{- printf "%s-neo4j" (include "project-brain.fullname" .) }}
{{- else }}
{{- .Values.externalNeo4j.host }}
{{- end }}
{{- end }}

{{/*
Redis hostname
*/}}
{{- define "project-brain.redis.host" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" (include "project-brain.fullname" .) }}
{{- else }}
{{- .Values.externalRedis.host }}
{{- end }}
{{- end }}

{{/*
PostgreSQL connection string
*/}}
{{- define "project-brain.postgresql.connectionString" -}}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.postgresql.global.postgresql.auth.username }}:{{ .Values.postgresql.global.postgresql.auth.password }}@{{ include "project-brain.postgresql.host" . }}:5432/{{ .Values.postgresql.global.postgresql.auth.database }}
{{- else }}
postgresql://{{ .Values.externalPostgresql.username }}:{{ .Values.externalPostgresql.password }}@{{ .Values.externalPostgresql.host }}:{{ .Values.externalPostgresql.port }}/{{ .Values.externalPostgresql.database }}
{{- end }}
{{- end }}

{{/*
Neo4j connection string
*/}}
{{- define "project-brain.neo4j.connectionString" -}}
{{- if .Values.neo4j.enabled }}
bolt://{{ include "project-brain.neo4j.host" . }}:7687
{{- else }}
bolt://{{ .Values.externalNeo4j.host }}:{{ .Values.externalNeo4j.port }}
{{- end }}
{{- end }}

{{/*
Redis connection string
*/}}
{{- define "project-brain.redis.connectionString" -}}
{{- if .Values.redis.enabled }}
redis://:{{ .Values.redis.auth.password }}@{{ include "project-brain.redis.host" . }}:6379/0
{{- else }}
redis://:{{ .Values.externalRedis.password }}@{{ .Values.externalRedis.host }}:{{ .Values.externalRedis.port }}/0
{{- end }}
{{- end }}

{{/*
API environment variables
*/}}
{{- define "project-brain.api.env" -}}
- name: ENVIRONMENT
  value: {{ .Values.global.environment | default "production" | quote }}
- name: LOG_LEVEL
  value: {{ .Values.api.env.LOG_LEVEL | default "INFO" | quote }}
- name: DATA_DIR
  value: {{ .Values.api.env.DATA_DIR | default "/app/data" | quote }}
- name: LOG_DIR
  value: {{ .Values.api.env.LOG_DIR | default "/app/logs" | quote }}

- name: POSTGRES_HOST
  value: {{ include "project-brain.postgresql.host" . | quote }}
- name: POSTGRES_PORT
  value: "5432"
- name: POSTGRES_DB
  value: {{ .Values.postgresql.global.postgresql.auth.database | quote }}
- name: POSTGRES_USER
  value: {{ .Values.postgresql.global.postgresql.auth.username | quote }}
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "project-brain.fullname" . }}-postgresql
      key: postgres-password

- name: NEO4J_URI
  value: {{ include "project-brain.neo4j.connectionString" . | quote }}
- name: NEO4J_USER
  value: "neo4j"
- name: NEO4J_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "project-brain.fullname" . }}-neo4j
      key: neo4j-password

- name: REDIS_URI
  value: {{ include "project-brain.redis.connectionString" . | quote }}

- name: JWT_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ include "project-brain.fullname" . }}-secrets
      key: jwt-secret

- name: API_KEY_ADMIN
  valueFrom:
    secretKeyRef:
      name: {{ include "project-brain.fullname" . }}-secrets
      key: api-key-admin

{{- with .Values.api.extraEnv }}
{{- toYaml . | nindent 0 }}
{{- end }}
{{- end }}

{{/*
API ConfigMap
*/}}
{{- define "project-brain.api.configmap" -}}
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "project-brain.fullname" . }}-config
  labels:
    {{- include "project-brain.labels" . | nindent 4 }}
data:
  system.yaml: |
{{ .Values.api.config.system.yaml | indent 4 }}
{{- end }}

{{/*
API Secret
*/}}
{{- define "project-brain.api.secret" -}}
apiVersion: v1
kind: Secret
metadata:
  name: {{ include "project-brain.fullname" . }}-secrets
  labels:
    {{- include "project-brain.labels" . | nindent 4 }}
type: Opaque
stringData:
  jwt-secret: {{ .Values.api.secrets.jwtSecret | default (randAlphaNum 32) | quote }}
  api-key-admin: {{ .Values.api.secrets.apiKeyAdmin | default (randAlphaNum 32) | quote }}
  api-key-service: {{ .Values.api.secrets.apiKeyService | default (randAlphaNum 32) | quote }}
{{- end }}
