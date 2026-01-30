"""
Script para verificar integridad de datos.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Iterable


# -------------------------------------------------
# UTILIDADES
# -------------------------------------------------

def base_result(**extra) -> Dict[str, Any]:
    return {
        "status": "passed",
        "details": [],
        **extra,
    }


def safe_load_json(path: Path) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def count_and_fail(results: Dict[str, Any], invalid_key: str) -> None:
    if results.get(invalid_key, 0) > 0:
        results["status"] = "failed"


# -------------------------------------------------
# CHECKER
# -------------------------------------------------

class DataIntegrityChecker:
    """Verificador de integridad de datos."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()

    # ----------------------------
    # EJECUCIÓN PRINCIPAL
    # ----------------------------

    def run_full_check(self) -> Dict[str, Any]:
        print("=== VERIFICACIÓN DE INTEGRIDAD DE DATOS ===")

        checks = {
            "projects": self._check_projects,
            "embeddings": self._check_embeddings,
            "graphs": self._check_graphs,
            "cache": self._check_cache,
            "state": self._check_state,
            "backups": self._check_backups,
        }

        results = {name: fn() for name, fn in checks.items()}
        results["overall"] = self._build_overall(results)

        self._print_summary(results["overall"])
        return results

    # ----------------------------
    # RESUMEN
    # ----------------------------

    def _build_overall(self, results: Dict[str, Any]) -> Dict[str, Any]:
        overall = {
            "passed": False,
            "total_checks": 0,
            "passed_checks": 0,
            "warnings": [],
            "errors": [],
        }

        for name, result in results.items():
            overall["total_checks"] += 1
            status = result.get("status", "failed")

            if status == "passed":
                overall["passed_checks"] += 1
            elif status == "warning":
                overall["warnings"].append(name)
            else:
                overall["errors"].append(name)

        overall["passed"] = (
            overall["passed_checks"] == overall["total_checks"]
        )
        return overall

    def _print_summary(self, overall: Dict[str, Any]) -> None:
        print("\n=== RESUMEN ===")
        print(f"Checks realizados: {overall['total_checks']}")
        print(f"Checks exitosos: {overall['passed_checks']}")
        print(f"Warnings: {len(overall['warnings'])}")
        print(f"Fallidos: {len(overall['errors'])}")

        print("✅ TODOS LOS CHECKS PASARON" if overall["passed"]
              else "❌ ALGUNOS CHECKS FALLARON")

    # -------------------------------------------------
    # CHECKS
    # -------------------------------------------------

    def _check_projects(self) -> Dict[str, Any]:
        print("\n[1/6] Verificando proyectos...")
        projects_dir = self.data_dir / "projects"

        results = base_result(
            projects_checked=0,
            valid_projects=0,
            invalid_projects=0,
        )

        if not projects_dir.exists():
            results["status"] = "failed"
            results["details"].append("Directorio de proyectos no existe")
            return results

        for project_dir in self._iter_dirs(projects_dir):
            results["projects_checked"] += 1
            check = self._check_single_project(project_dir)

            if check["valid"]:
                results["valid_projects"] += 1
            else:
                results["invalid_projects"] += 1
                results["details"].append(
                    f"{project_dir.name}: {check['error']}"
                )

        count_and_fail(results, "invalid_projects")
        return results

    def _check_single_project(self, project_dir: Path) -> Dict[str, Any]:
        for name in ("metadata.json", "files.json"):
            if not (project_dir / name).exists():
                return {"valid": False, "error": f"Falta {name}"}

        metadata = safe_load_json(project_dir / "metadata.json")
        if not metadata or "id" not in metadata:
            return {"valid": False, "error": "metadata.json inválido o sin id"}

        return {"valid": True}

    # ----------------------------

    def _check_embeddings(self) -> Dict[str, Any]:
        print("[2/6] Verificando embeddings...")
        embeddings_dir = self.data_dir / "embeddings"

        results = base_result(
            embeddings_checked=0,
            valid_embeddings=0,
            invalid_embeddings=0,
        )

        if not embeddings_dir.exists():
            results["status"] = "warning"
            results["details"].append("Directorio de embeddings no existe")
            return results

        self._check_chroma(embeddings_dir, results)
        self._check_embedding_cache(embeddings_dir, results)

        count_and_fail(results, "invalid_embeddings")
        return results

    def _check_chroma(self, embeddings_dir: Path, results: Dict[str, Any]) -> None:
        chroma_dir = embeddings_dir / "chroma"
        if not chroma_dir.exists():
            return

        for fname in ("chroma.sqlite3", "chroma_settings.json"):
            if not (chroma_dir / fname).exists():
                results["status"] = "failed"
                results["details"].append(f"Falta archivo ChromaDB: {fname}")

    def _check_embedding_cache(self, embeddings_dir: Path, results: Dict[str, Any]) -> None:
        cache_dir = embeddings_dir / "cache"
        if not cache_dir.exists():
            return

        for cache_file in cache_dir.rglob("*.json"):
            results["embeddings_checked"] += 1
            data = safe_load_json(cache_file)
            if data and "embedding" in data and "metadata" in data:
                results["valid_embeddings"] += 1
            else:
                results["invalid_embeddings"] += 1

    # ----------------------------

    def _check_graphs(self) -> Dict[str, Any]:
        print("[3/6] Verificando grafos...")
        graphs_dir = self.data_dir / "graph_exports"

        results = base_result(
            graphs_checked=0,
            valid_graphs=0,
            invalid_graphs=0,
        )

        if not graphs_dir.exists():
            results["status"] = "warning"
            results["details"].append("Directorio de grafos no existe")
            return results

        for graph_file in graphs_dir.glob("*"):
            if graph_file.suffix not in {".json", ".graphml", ".gexf"}:
                continue

            results["graphs_checked"] += 1
            if self._is_valid_graph(graph_file):
                results["valid_graphs"] += 1
            else:
                results["invalid_graphs"] += 1

        count_and_fail(results, "invalid_graphs")
        return results

    def _is_valid_graph(self, graph_file: Path) -> bool:
        try:
            if graph_file.suffix == ".json":
                data = safe_load_json(graph_file)
                return bool(data and "nodes" in data and "edges" in data)

            with open(graph_file, "r", encoding="utf-8", errors="ignore") as f:
                f.read(256)
            return True
        except Exception:
            return False

    # ----------------------------

    def _check_cache(self) -> Dict[str, Any]:
        print("[4/6] Verificando caché...")
        cache_dir = self.data_dir / "cache"

        results = base_result(
            cache_entries_checked=0,
            valid_entries=0,
            invalid_entries=0,
        )

        if not cache_dir.exists():
            results["status"] = "warning"
            results["details"].append("Directorio de caché no existe")
            return results

        disk_cache = cache_dir / "l3_disk"
        if not disk_cache.exists():
            return results

        import pickle

        for cache_file in disk_cache.glob("*.pkl"):
            results["cache_entries_checked"] += 1
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "value" in data:
                    results["valid_entries"] += 1
                else:
                    results["invalid_entries"] += 1
            except Exception:
                results["invalid_entries"] += 1

        count_and_fail(results, "invalid_entries")
        return results

    # ----------------------------

    def _check_state(self) -> Dict[str, Any]:
        print("[5/6] Verificando estado...")
        state_dir = self.data_dir / "state"

        results = base_result(
            components_checked=0,
            valid_components=0,
            invalid_components=0,
        )

        if not state_dir.exists():
            results["status"] = "warning"
            results["details"].append("Directorio de estado no existe")
            return results

        db_file = state_dir / "state.db"
        if not db_file.exists():
            return results

        import sqlite3

        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            for table in ("sessions", "operations", "components", "workflows"):
                cursor.execute(f"SELECT 1 FROM {table} LIMIT 1")
                results["components_checked"] += 1
                results["valid_components"] += 1

        except Exception as e:
            results["status"] = "failed"
            results["details"].append(str(e))
        finally:
            try:
                conn.close()
            except Exception:
                pass

        return results

    # ----------------------------

    def _check_backups(self) -> Dict[str, Any]:
        print("[6/6] Verificando backups...")
        backups_dir = self.data_dir / "backups"

        results = base_result()

        if not backups_dir.exists():
            results["status"] = "warning"
            results["details"].append("Directorio de backups no existe")
            return results

        metadata_file = backups_dir / "backup_metadata.json"
        if not metadata_file.exists():
            return results

        metadata = safe_load_json(metadata_file)
        if not metadata or "backups" not in metadata:
            results["status"] = "failed"

        return results

    # -------------------------------------------------

    @staticmethod
    def _iter_dirs(path: Path) -> Iterable[Path]:
        return (p for p in path.iterdir() if p.is_dir())


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main() -> int:
    data_dir = Path("./data")

    if not data_dir.exists():
        print("❌ Directorio de datos no encontrado")
        return 1

    checker = DataIntegrityChecker(data_dir)
    results = checker.run_full_check()

    output_file = data_dir / "integrity_check.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Resultados guardados en: {output_file}")
    return 0 if results["overall"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
