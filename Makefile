# Biomedical Knowledge Graph Pipeline
#
# Quick start:
#   make quick           # Fast test (100 abstracts, ~5 min)
#   make build           # Full build + Neo4j (20K abstracts, ~2 hours)
#   make demo SEARCH_TERM="diabetes"  # Custom search demo
#   make continue DIR=data/...  # Resume from existing directory

SHELL := /bin/bash
.DEFAULT_GOAL := help

#=============================================================================
# Help & Discovery
#=============================================================================

.PHONY: help
help: ## Show this help message
	@echo "Biomedical Knowledge Graph Pipeline"
	@echo ""
	@echo "Main workflows (via Python CLI):"
	@echo "  make quick          # Fast test (100 abstracts, ~5 min)"
	@echo "  make build          # Full build + Neo4j (20K abstracts, ~2 hours)"
	@echo "  make demo SEARCH_TERM=\"...\" # Custom search demo (100 abstracts, ~5 min)"
	@echo ""
	@echo "Continue existing:"
	@echo "  make continue DIR=data/...  # Resume from directory"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          # Remove generated data"
	@echo ""
	@echo "All targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

#=============================================================================
# Pipeline Commands (thin wrappers around Python CLI)
#=============================================================================

.PHONY: quick
quick: ## Fast pipeline test (100 abstracts, ~5 min)
	poetry run python -m biomed_kg_agent.cli run-pipeline --preset quick

.PHONY: build
build: ## Full pipeline with Neo4j migration (20K abstracts, ~2 hours)
	poetry run python -m biomed_kg_agent.cli run-pipeline --preset full --migrate --clear-neo4j

.PHONY: demo
demo: ## Custom search demo (usage: make demo SEARCH_TERM="your search")
	@if [ -z "$(SEARCH_TERM)" ]; then \
		echo "Error: SEARCH_TERM required"; \
		echo "Usage: make demo SEARCH_TERM=\"diabetes\""; \
		exit 1; \
	fi
	poetry run python -m biomed_kg_agent.cli run-pipeline \
		--search-term "$(SEARCH_TERM)" \
		--size 100

.PHONY: continue
continue: ## Continue from existing directory (usage: make continue DIR=data/...)
	poetry run python -m biomed_kg_agent.cli continue-pipeline --dir $(DIR) $(if $(MIGRATE),--migrate --clear-neo4j)

#=============================================================================
# Neo4j Migration
#=============================================================================

.PHONY: migrate
migrate: ## Migrate to Neo4j (usage: make migrate DIR=data/...)
	poetry run python -m biomed_kg_agent.cli migrate-to-neo4j --input $(DIR)/kg.db --clear-existing

#=============================================================================
# Development & Testing
#=============================================================================

.PHONY: test
test: ## Run pytest and code quality checks
	@echo "Running tests..."
	poetry run pytest -q
	@echo "Running code quality checks..."
	poetry run pre-commit run --all-files

#=============================================================================
# Cleanup
#=============================================================================

.PHONY: clean
clean: ## Remove generated pipeline data
	@echo "Cleaning..."
	rm -rf data/quick_* data/breast_cancer_targeted_therapy_* data/custom_*
	@echo "Done"

.PHONY: clean-all
clean-all: ## Deep clean: all data dirs and logs
	@echo "Deep cleaning..."
	rm -rf data/*/ logs/*.log
	@echo "Done"
