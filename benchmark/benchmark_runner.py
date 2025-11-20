"""
Benchmark Runner for Multi-Agent RAG System Evaluation.

This module executes the MAS against a test query dataset and collects
comprehensive scientific metrics for evaluation and visualization.

Supports 3 search types: vector, graph, and hybrid.

Usage:
    python benchmark_runner.py --queries test_queries.json --output results.json
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class SearchType(str, Enum):
    """Search type options."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class MetricCategory(str, Enum):
    """Metric categories."""
    RETRIEVAL = "retrieval"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    PERFORMANCE = "performance"


@dataclass
class QueryResult:
    """Result of a single query execution."""
    query_id: str
    query_text: str
    category: str
    complexity: str
    search_type: str  # ✅ NEW: vector, graph, or hybrid
    
    # Retrieval metrics
    papers_retrieved: int
    expected_papers: int
    retrieval_precision: float
    papers_cited: int  # ✅ NEW: Papers actually cited in answer
    
    # Validation metrics
    verified_claims: int
    total_claims: int
    hallucination_score: float
    citation_accuracy: float
    
    # Synthesis metrics
    answer_length: int
    answer_confidence: float
    
    # Performance metrics
    total_latency_ms: float
    retrieval_latency_ms: float
    synthesis_latency_ms: float
    validation_latency_ms: float
    
    # Metadata
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Overall benchmark report."""
    benchmark_id: str
    dataset_name: str
    total_queries: int
    completed_queries: int
    failed_queries: int
    start_time: str
    end_time: str
    duration_seconds: float
    
    # Aggregated metrics
    avg_retrieval_precision: float
    avg_hallucination_score: float
    avg_citation_accuracy: float
    avg_answer_confidence: float
    avg_total_latency_ms: float
    
    # Per-category metrics
    category_metrics: Dict[str, Dict[str, float]]
    complexity_metrics: Dict[str, Dict[str, float]]
    search_type_metrics: Dict[str, Dict[str, float]]  # ✅ NEW: Per-search-type metrics
    
    # Query results
    query_results: List[QueryResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['query_results'] = [qr.to_dict() for qr in self.query_results]
        return data


# ============================================================================
# BENCHMARK EXECUTOR
# ============================================================================

class BenchmarkExecutor:
    """Executes benchmark on multi-agent RAG system."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8058",
        timeout_seconds: int = 120
    ):
        """
        Initialize benchmark executor.
        
        Args:
            api_url: URL of the RAG system API
            timeout_seconds: Timeout for each query
        """
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.benchmark_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Initialized BenchmarkExecutor (ID: {self.benchmark_id})")
    
    async def execute_query(
        self,
        query_id: str,
        query_text: str,
        expected_papers: List[str],
        ground_truth_claims: List[str],
        query_metadata: Dict[str, Any]
    ) -> QueryResult:
        """
        Execute a single query and collect metrics.
        
        Args:
            query_id: Unique query identifier
            query_text: Query text to submit
            expected_papers: Papers expected in results
            ground_truth_claims: Claims expected to be verified
            query_metadata: Additional query metadata (includes search_type)
        
        Returns:
            QueryResult with all collected metrics
        """
        search_type = query_metadata.get("search_type", "hybrid")
        logger.info(f"Executing query {query_id} [{search_type}]: {query_text[:60]}...")
        
        start_time = time.time()
        
        try:
            # Make API request
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.api_url}/chat/mas",
                    json={
                        "message": query_text,
                        "search_type": search_type,  # ✅ Use from query metadata
                        "enable_validation": True
                    }
                )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")
            
            result_json = response.json()
            end_time = time.time()
            total_latency_ms = (end_time - start_time) * 1000
            
            # Extract metrics
            metrics = self._extract_metrics(
                result_json,
                query_id,
                query_text,
                expected_papers,
                ground_truth_claims,
                query_metadata,
                total_latency_ms
            )
            
            logger.info(
                f"Query {query_id} [{search_type}] complete: "
                f"papers={metrics.papers_retrieved}, "
                f"hallucination={metrics.hallucination_score:.2f}, "
                f"latency={metrics.total_latency_ms:.0f}ms"
            )
            
            return metrics
        
        except Exception as e:
            logger.error(f"Query {query_id} [{search_type}] failed: {e}")
            
            return QueryResult(
                query_id=query_id,
                query_text=query_text,
                category=query_metadata.get("category", "unknown"),
                complexity=query_metadata.get("complexity", "unknown"),
                search_type=search_type,
                papers_retrieved=0,
                expected_papers=len(expected_papers),
                retrieval_precision=0.0,
                papers_cited=0,
                verified_claims=0,
                total_claims=0,
                hallucination_score=1.0,
                citation_accuracy=0.0,
                answer_length=0,
                answer_confidence=0.0,
                total_latency_ms=(time.time() - start_time) * 1000,
                retrieval_latency_ms=0.0,
                synthesis_latency_ms=0.0,
                validation_latency_ms=0.0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
    
    def _extract_metrics(
        self,
        result_json: Dict[str, Any],
        query_id: str,
        query_text: str,
        expected_papers: List[str],
        ground_truth_claims: List[str],
        query_metadata: Dict[str, Any],
        total_latency_ms: float
    ) -> QueryResult:
        """Extract metrics from API response."""
        # Retrieval metrics
        sources = result_json.get("sources", [])
        papers_retrieved = len(sources)
        
        # ✅ Calculate retrieval precision based on expected papers
        if papers_retrieved > 0:
            matching_papers = [
                s for s in sources 
                if any(ep.lower() in s.lower() for ep in expected_papers)
            ]
            retrieval_precision = len(matching_papers) / papers_retrieved
        else:
            retrieval_precision = 0.0
        
        # Validation metrics
        validation = result_json.get("validation", {})
        verified_claims = validation.get("verified_claims", 0)
        total_claims = validation.get("total_claims", 0)
        hallucination_score = validation.get("hallucination_score", 0.0)
        
        # Citation accuracy
        citation_verification = validation.get("citation_verification", {})
        citation_accuracy = citation_verification.get("citation_accuracy", 0.0)
        papers_cited = validation.get("papers_referenced", 0)
        
        # Synthesis metrics
        message = result_json.get("message", "")
        answer_length = len(message)
        answer_confidence = result_json.get("confidence", 0.0)
        
        # Performance metrics
        metadata = result_json.get("metadata", {})
        retrieval_latency_ms = metadata.get("retrieval_latency_ms", 0.0)
        synthesis_latency_ms = metadata.get("synthesis_latency_ms", 0.0)
        validation_latency_ms = metadata.get("validation_latency_ms", 0.0)
        
        # If component latencies are 0, estimate from total
        if retrieval_latency_ms == 0 and synthesis_latency_ms == 0 and validation_latency_ms == 0:
            # Estimate: 20% retrieval, 30% synthesis, 50% validation
            retrieval_latency_ms = total_latency_ms * 0.2
            synthesis_latency_ms = total_latency_ms * 0.3
            validation_latency_ms = total_latency_ms * 0.5
        
        return QueryResult(
            query_id=query_id,
            query_text=query_text,
            category=query_metadata.get("category", "unknown"),
            complexity=query_metadata.get("complexity", "unknown"),
            search_type=query_metadata.get("search_type", "hybrid"),
            papers_retrieved=papers_retrieved,
            expected_papers=len(expected_papers),
            retrieval_precision=retrieval_precision,
            papers_cited=papers_cited,
            verified_claims=verified_claims,
            total_claims=total_claims,
            hallucination_score=hallucination_score,
            citation_accuracy=citation_accuracy,
            answer_length=answer_length,
            answer_confidence=answer_confidence,
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            synthesis_latency_ms=synthesis_latency_ms,
            validation_latency_ms=validation_latency_ms,
            timestamp=datetime.now().isoformat(),
            success=True
        )
    
    async def run_benchmark(
        self,
        test_queries_file: str,
        max_parallel: int = 3
    ) -> BenchmarkReport:
        """
        Run complete benchmark on all test queries.
        
        Args:
            test_queries_file: Path to test_queries.json
            max_parallel: Maximum parallel queries
        
        Returns:
            BenchmarkReport with all results
        """
        logger.info(f"Loading test queries from {test_queries_file}")
        
        with open(test_queries_file) as f:
            data = json.load(f)
        
        test_queries = data["test_queries"]
        total_queries = len(test_queries)
        
        logger.info(f"Loaded {total_queries} test queries")
        
        # Count by search type
        search_type_counts = {}
        for q in test_queries:
            st = q.get("search_type", "hybrid")
            search_type_counts[st] = search_type_counts.get(st, 0) + 1
        
        logger.info(f"Search type distribution: {search_type_counts}")
        
        # Execute queries with parallelization
        start_time = time.time()
        query_results = []
        
        for i in range(0, total_queries, max_parallel):
            batch = test_queries[i:i + max_parallel]
            
            tasks = [
                self.execute_query(
                    query_id=q["id"],
                    query_text=q["query"],
                    expected_papers=q.get("expected_papers", []),
                    ground_truth_claims=q.get("ground_truth_claims", []),
                    query_metadata={
                        "category": q.get("category"),
                        "complexity": q.get("complexity"),
                        "search_type": q.get("search_type", "hybrid")  # ✅ Pass search_type
                    }
                )
                for q in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            query_results.extend(batch_results)
            
            completed = len(query_results)
            logger.info(
                f"Progress: {completed}/{total_queries} "
                f"({completed/total_queries:.1%})"
            )
        
        end_time = time.time()
        
        # Compute aggregated metrics
        successful_results = [qr for qr in query_results if qr.success]
        failed_results = [qr for qr in query_results if not qr.success]
        
        report = self._create_report(
            test_queries_file,
            query_results,
            successful_results,
            failed_results,
            start_time,
            end_time
        )
        
        logger.info(
            f"Benchmark complete: {len(successful_results)}/{total_queries} successful, "
            f"avg hallucination={report.avg_hallucination_score:.2%}, "
            f"avg latency={report.avg_total_latency_ms:.0f}ms"
        )
        
        return report
    
    def _create_report(
        self,
        dataset_name: str,
        all_results: List[QueryResult],
        successful_results: List[QueryResult],
        failed_results: List[QueryResult],
        start_time: float,
        end_time: float
    ) -> BenchmarkReport:
        """Create benchmark report from results."""
        if not successful_results:
            avg_retrieval_precision = 0.0
            avg_hallucination_score = 1.0
            avg_citation_accuracy = 0.0
            avg_answer_confidence = 0.0
            avg_total_latency_ms = 0.0
        else:
            avg_retrieval_precision = np.mean([r.retrieval_precision for r in successful_results])
            avg_hallucination_score = np.mean([r.hallucination_score for r in successful_results])
            avg_citation_accuracy = np.mean([r.citation_accuracy for r in successful_results])
            avg_answer_confidence = np.mean([r.answer_confidence for r in successful_results])
            avg_total_latency_ms = np.mean([r.total_latency_ms for r in successful_results])
        
        # Per-category metrics
        categories = set(r.category for r in successful_results)
        category_metrics = {}
        for category in categories:
            cat_results = [r for r in successful_results if r.category == category]
            category_metrics[category] = {
                "count": len(cat_results),
                "avg_hallucination": float(np.mean([r.hallucination_score for r in cat_results])),
                "avg_citation_accuracy": float(np.mean([r.citation_accuracy for r in cat_results])),
                "avg_latency_ms": float(np.mean([r.total_latency_ms for r in cat_results]))
            }
        
        # Per-complexity metrics
        complexities = set(r.complexity for r in successful_results)
        complexity_metrics = {}
        for complexity in complexities:
            cpx_results = [r for r in successful_results if r.complexity == complexity]
            complexity_metrics[complexity] = {
                "count": len(cpx_results),
                "avg_hallucination": float(np.mean([r.hallucination_score for r in cpx_results])),
                "avg_citation_accuracy": float(np.mean([r.citation_accuracy for r in cpx_results])),
                "avg_latency_ms": float(np.mean([r.total_latency_ms for r in cpx_results]))
            }
        
        # ✅ NEW: Per-search-type metrics
        search_types = set(r.search_type for r in successful_results)
        search_type_metrics = {}
        for search_type in search_types:
            st_results = [r for r in successful_results if r.search_type == search_type]
            search_type_metrics[search_type] = {
                "count": len(st_results),
                "avg_hallucination": float(np.mean([r.hallucination_score for r in st_results])),
                "avg_citation_accuracy": float(np.mean([r.citation_accuracy for r in st_results])),
                "avg_retrieval_precision": float(np.mean([r.retrieval_precision for r in st_results])),
                "avg_latency_ms": float(np.mean([r.total_latency_ms for r in st_results])),
                "avg_papers_cited": float(np.mean([r.papers_cited for r in st_results]))
            }
        
        return BenchmarkReport(
            benchmark_id=self.benchmark_id,
            dataset_name=Path(dataset_name).stem,
            total_queries=len(all_results),
            completed_queries=len(successful_results),
            failed_queries=len(failed_results),
            start_time=datetime.fromtimestamp(start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            duration_seconds=end_time - start_time,
            avg_retrieval_precision=avg_retrieval_precision,
            avg_hallucination_score=avg_hallucination_score,
            avg_citation_accuracy=avg_citation_accuracy,
            avg_answer_confidence=avg_answer_confidence,
            avg_total_latency_ms=avg_total_latency_ms,
            category_metrics=category_metrics,
            complexity_metrics=complexity_metrics,
            search_type_metrics=search_type_metrics,  # ✅ NEW
            query_results=all_results
        )


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark Multi-Agent RAG System (Vector, Graph, Hybrid)"
    )
    parser.add_argument(
        "--queries",
        default="test_queries.json",
        help="Path to test queries JSON file"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file for benchmark results"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8058",
        help="URL of the RAG system API"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Maximum parallel queries"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per query (seconds)"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    executor = BenchmarkExecutor(
        api_url=args.api_url,
        timeout_seconds=args.timeout
    )
    report = await executor.run_benchmark(
        test_queries_file=args.queries,
        max_parallel=args.max_parallel
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK REPORT")
    print("="*80)
    print(f"Benchmark ID: {report.benchmark_id}")
    print(f"Dataset: {report.dataset_name}")
    print(f"Total Queries: {report.total_queries}")
    print(f"Completed: {report.completed_queries}")
    print(f"Failed: {report.failed_queries}")
    print(f"Duration: {report.duration_seconds:.1f}s")
    print()
    print("Aggregated Metrics:")
    print(f"  Retrieval Precision: {report.avg_retrieval_precision:.2%}")
    print(f"  Hallucination Score: {report.avg_hallucination_score:.2%}")
    print(f"  Citation Accuracy: {report.avg_citation_accuracy:.2%}")
    print(f"  Average Confidence: {report.avg_answer_confidence:.2f}")
    print(f"  Average Latency: {report.avg_total_latency_ms:.0f}ms")
    print()
    
    # ✅ NEW: Search type comparison
    if report.search_type_metrics:
        print("Search Type Comparison:")
        print(f"  {'Type':<10} {'Halluc':>8} {'Citation':>9} {'Precision':>10} {'Latency':>9}")
        print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*10} {'-'*9}")
        for st, metrics in sorted(report.search_type_metrics.items()):
            print(
                f"  {st:<10} "
                f"{metrics['avg_hallucination']:>7.1%} "
                f"{metrics['avg_citation_accuracy']:>8.1%} "
                f"{metrics['avg_retrieval_precision']:>9.1%} "
                f"{metrics['avg_latency_ms']:>8.0f}ms"
            )
    
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
