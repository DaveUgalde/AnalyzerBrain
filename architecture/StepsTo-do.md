# Crear estructura principal
mkdir -p .github/workflows
mkdir -p .vscode
mkdir -p architecture
mkdir -p config
mkdir -p data/{backups,cache,embeddings,graph_exports,projects,state}
mkdir -p deployments/{docker,helm/templates,kubernetes}
mkdir -p docs/{api,architecture,deployment,developer,examples,user_guide}
mkdir -p logs
mkdir -p monitoring/{alerts,grafana/{dashboards,datasources},loki,prometheus}
mkdir -p requirements
mkdir -p scripts
mkdir -p src/{api,agents,core,embeddings,graph,indexer,learning,memory,utils}
mkdir -p tests/{analyzer_code,e2e,fixtures/{sample_code,sample_project},integration,performance,unit}