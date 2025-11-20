"""
Result aggregation utilities for Multi-Agent System.

This module provides functions to merge and aggregate results from multiple agents
for scientific literature analysis:
- Combine retrieval results (deduplication, ranking)
- Merge knowledge graph data (entities, relationships)
- Aggregate synthesis results
- Combine validation reports
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# RETRIEVAL RESULT AGGREGATION
# ============================================================================

class RetrievalAggregator:
    """Aggregate and deduplicate retrieval results for scientific papers."""
    
    @staticmethod
    def aggregate_chunks(
        retrieval_results: List[Dict[str, Any]],
        max_chunks: int = 20,
        dedup_threshold: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Aggregate paper chunks from multiple retrieval results.
        
        Args:
            retrieval_results: List of retrieval result dicts
            max_chunks: Maximum chunks to return
            dedup_threshold: Similarity threshold for deduplication
        
        Returns:
            Aggregated and deduplicated chunks
        """
        all_chunks = []
        
        # Extract all chunks
        for result in retrieval_results:
            if isinstance(result, dict) and 'data' in result:
                result_data = result['data']
                chunks = result_data.get('chunks', [])
                all_chunks.extend(chunks)
        
        logger.info(
            f"Aggregating {len(all_chunks)} paper chunks from "
            f"{len(retrieval_results)} results"
        )
        
        # Deduplicate by chunk_id
        seen_ids = set()
        unique_chunks = []
        
        for chunk in all_chunks:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)
        
        logger.info(f"After deduplication: {len(unique_chunks)} unique chunks")
        
        # Sort by score (descending)
        unique_chunks.sort(key=lambda c: c.get('score', 0.0), reverse=True)
        
        # Limit to max_chunks
        final_chunks = unique_chunks[:max_chunks]
        
        logger.info(f"Returning top {len(final_chunks)} paper chunks")
        
        return final_chunks
    
    @staticmethod
    def aggregate_graph_facts(
        retrieval_results: List[Dict[str, Any]],
        max_facts: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Aggregate knowledge graph facts from multiple results.
        
        Args:
            retrieval_results: List of retrieval result dicts
            max_facts: Maximum facts to return
        
        Returns:
            Aggregated facts
        """
        all_facts = []
        
        # Extract all facts
        for result in retrieval_results:
            if isinstance(result, dict) and 'data' in result:
                result_data = result['data']
                facts = result_data.get('graph_facts', [])
                all_facts.extend(facts)
        
        logger.info(f"Aggregating {len(all_facts)} graph facts")
        
        # Deduplicate by UUID
        seen_uuids = set()
        unique_facts = []
        
        for fact in all_facts:
            fact_uuid = fact.get('uuid')
            if fact_uuid and fact_uuid not in seen_uuids:
                seen_uuids.add(fact_uuid)
                unique_facts.append(fact)
        
        # If no UUID, deduplicate by fact text
        if not unique_facts:
            seen_texts = set()
            for fact in all_facts:
                fact_text = fact.get('fact', str(fact))
                if fact_text not in seen_texts:
                    seen_texts.add(fact_text)
                    unique_facts.append(fact)
        
        logger.info(f"After deduplication: {len(unique_facts)} unique facts")
        
        # Limit to max_facts
        final_facts = unique_facts[:max_facts]
        
        return final_facts
    
    @staticmethod
    def aggregate_sources(
        retrieval_results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Aggregate unique paper sources from retrieval results.
        
        Args:
            retrieval_results: List of retrieval result dicts
        
        Returns:
            List of unique paper source names
        """
        all_sources = set()
        
        for result in retrieval_results:
            if isinstance(result, dict) and 'data' in result:
                result_data = result['data']
                sources = result_data.get('sources', [])
                all_sources.update(sources)
        
        return sorted(list(all_sources))
    
    @staticmethod
    def calculate_aggregate_metrics(
        retrieval_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate aggregate retrieval metrics.
        
        Args:
            retrieval_results: List of retrieval result dicts
        
        Returns:
            Aggregate metrics
        """
        if not retrieval_results:
            return {
                "avg_score": 0.0,
                "total_results": 0,
                "unique_sources": 0,
                "unique_papers": 0
            }
        
        # Extract scores
        scores = []
        total_results = 0
        unique_papers = set()
        
        for result in retrieval_results:
            if isinstance(result, dict) and 'data' in result:
                result_data = result['data']
                
                # Get chunks with scores
                chunks = result_data.get('chunks', [])
                for chunk in chunks:
                    score = chunk.get('score')
                    if score is not None:
                        scores.append(score)
                    
                    # Track unique papers
                    paper_id = chunk.get('document_id') or chunk.get('document_title')
                    if paper_id:
                        unique_papers.add(paper_id)
                
                total_results += result_data.get('total_results', 0)
        
        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Get unique sources
        sources = RetrievalAggregator.aggregate_sources(retrieval_results)
        
        return {
            "avg_score": avg_score,
            "total_results": total_results,
            "unique_sources": len(sources),
            "unique_papers": len(unique_papers),
            "results_with_scores": len(scores)
        }


# ============================================================================
# KNOWLEDGE GRAPH AGGREGATION
# ============================================================================

class KnowledgeAggregator:
    """Aggregate knowledge graph data for academic entities."""
    
    @staticmethod
    def aggregate_entities(
        knowledge_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Aggregate academic entities from multiple knowledge results.
        
        Args:
            knowledge_results: List of knowledge result dicts
        
        Returns:
            Aggregated entities (authors, papers, venues, etc.)
        """
        entity_map = {}  # name -> entity data
        
        # Collect all entities
        for result in knowledge_results:
            if isinstance(result, dict) and 'data' in result:
                result_data = result['data']
                entities = result_data.get('entities', [])
                
                for entity in entities:
                    name = entity.get('name')
                    if name:
                        if name not in entity_map:
                            entity_map[name] = entity
                        else:
                            # Merge entity data (enrich existing)
                            existing = entity_map[name]
                            
                            # Merge related entities
                            existing_related = set(existing.get('related_entities', []))
                            new_related = entity.get('related_entities', [])
                            existing_related.update(new_related)
                            existing['related_entities'] = list(existing_related)
                            
                            # Merge facts
                            existing_facts = existing.get('facts', [])
                            new_facts = entity.get('facts', [])
                            existing['facts'] = existing_facts + new_facts
        
        entities = list(entity_map.values())
        
        logger.info(f"Aggregated {len(entities)} unique academic entities")
        
        return entities
    
    @staticmethod
    def aggregate_relationships(
        knowledge_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Aggregate relationships from multiple knowledge results (citations, authorship, etc.).
        
        Args:
            knowledge_results: List of knowledge result dicts
        
        Returns:
            Aggregated relationships
        """
        relationship_set = set()
        relationships = []
        
        # Collect all relationships
        for result in knowledge_results:
            if isinstance(result, dict) and 'data' in result:
                result_data = result['data']
                rels = result_data.get('relationships', [])
                
                for rel in rels:
                    source = rel.get('source', '')
                    target = rel.get('target', '')
                    rel_type = rel.get('type', '')
                    
                    # Create unique key
                    key = (source, target, rel_type)
                    
                    if key not in relationship_set:
                        relationship_set.add(key)
                        relationships.append(rel)
        
        logger.info(f"Aggregated {len(relationships)} unique relationships")
        
        return relationships
    
    @staticmethod
    def aggregate_temporal_facts(
        knowledge_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Aggregate temporal facts and create unified timeline (paper publication dates, etc.).
        
        Args:
            knowledge_results: List of knowledge result dicts
        
        Returns:
            Sorted timeline events
        """
        all_events = []
        
        # Collect all temporal facts
        for result in knowledge_results:
            if isinstance(result, dict) and 'data' in result:
                result_data = result['data']
                timeline = result_data.get('timeline', [])
                all_events.extend(timeline)
        
        # Sort by date
        def get_date(event):
            date = event.get('date') or event.get('valid_at') or ''
            return date
        
        sorted_events = sorted(all_events, key=get_date)
        
        # Deduplicate by fact text
        seen_facts = set()
        unique_events = []
        
        for event in sorted_events:
            fact = event.get('event') or event.get('fact') or str(event)
            if fact not in seen_facts:
                seen_facts.add(fact)
                unique_events.append(event)
        
        logger.info(f"Aggregated {len(unique_events)} unique temporal facts")
        
        return unique_events
    
    @staticmethod
    def build_knowledge_graph(
        knowledge_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build unified knowledge graph from multiple results.
        
        Args:
            knowledge_results: List of knowledge result dicts
        
        Returns:
            Unified knowledge graph
        """
        entities = KnowledgeAggregator.aggregate_entities(knowledge_results)
        relationships = KnowledgeAggregator.aggregate_relationships(knowledge_results)
        timeline = KnowledgeAggregator.aggregate_temporal_facts(knowledge_results)
        
        # Create entity index
        entity_index = {e['name']: e for e in entities}
        
        # Create relationship graph
        graph = defaultdict(list)
        for rel in relationships:
            source = rel.get('source')
            if source:
                graph[source].append(rel)
        
        return {
            "entities": entities,
            "relationships": relationships,
            "timeline": timeline,
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "timeline_event_count": len(timeline),
            "entity_index": entity_index,
            "relationship_graph": dict(graph)
        }


# ============================================================================
# SYNTHESIS RESULT AGGREGATION
# ============================================================================

class SynthesisAggregator:
    """Aggregate synthesis results for research answers."""
    
    @staticmethod
    def merge_synthesis_results(
        synthesis_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge multiple synthesis results into one.
        
        Args:
            synthesis_results: List of synthesis result dicts
        
        Returns:
            Merged synthesis result with citations
        """
        if not synthesis_results:
            return {
                "answer": "",
                "confidence": 0.0,
                "sources_used": [],
                "reasoning": "",
                "papers_cited": 0
            }
        
        # If only one result, return it
        if len(synthesis_results) == 1:
            result = synthesis_results[0]
            if isinstance(result, dict) and 'data' in result:
                return result['data']
            return result
        
        # Multiple results - choose best or combine
        logger.info(f"Merging {len(synthesis_results)} synthesis results")
        
        # Choose result with highest confidence
        best_result = None
        best_confidence = -1.0
        
        for result in synthesis_results:
            if isinstance(result, dict) and 'data' in result:
                result_data = result['data']
                confidence = result_data.get('confidence', 0.0)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = result_data
        
        # Aggregate sources from all results
        all_sources = set()
        for result in synthesis_results:
            if isinstance(result, dict) and 'data' in result:
                result_data = result['data']
                sources = result_data.get('sources_used', [])
                all_sources.update(sources)
        
        if best_result:
            best_result['sources_used'] = list(all_sources)
        
        return best_result or synthesis_results[0]
    
    @staticmethod
    def combine_answers(
        synthesis_results: List[Dict[str, Any]],
        separator: str = "\n\n"
    ) -> str:
        """
        Combine multiple answers into one.
        
        Args:
            synthesis_results: List of synthesis result dicts
            separator: Separator between answers
        
        Returns:
            Combined answer text
        """
        answers = []
        
        for result in synthesis_results:
            if isinstance(result, dict) and 'data' in result:
                result_data = result['data']
                answer = result_data.get('answer', '')
                if answer:
                    answers.append(answer)
        
        return separator.join(answers)


# ============================================================================
# VALIDATION RESULT AGGREGATION
# ============================================================================

class ValidationAggregator:
    """Aggregate validation results for citation verification."""
    
    @staticmethod
    def merge_validation_reports(
        validation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge multiple validation reports.
        
        Args:
            validation_results: List of validation result dicts
        
        Returns:
            Merged validation report
        """
        if not validation_results:
            return {
                "is_valid": False,
                "fact_checks": [],
                "hallucination_score": 1.0,
                "citations": [],
                "issues": ["No validation performed"],
                "papers_referenced": 0
            }
        
        # If only one result, return it
        if len(validation_results) == 1:
            result = validation_results[0]
            if isinstance(result, dict) and 'data' in result:
                return result['data']
            return result
        
        logger.info(f"Merging {len(validation_results)} validation reports")
        
        # Aggregate all fact checks
        all_fact_checks = []
        all_citations = set()
        all_issues = []
        hallucination_scores = []
        papers_referenced = set()
        
        for result in validation_results:
            if isinstance(result, dict) and 'data' in result:
                result_data = result['data']
                
                fact_checks = result_data.get('fact_checks', [])
                all_fact_checks.extend(fact_checks)
                
                citations = result_data.get('citations', [])
                all_citations.update(citations)
                
                issues = result_data.get('issues', [])
                all_issues.extend(issues)
                
                score = result_data.get('hallucination_score', 0.0)
                hallucination_scores.append(score)
                
                papers_ref = result_data.get('papers_referenced', 0)
                if papers_ref > 0:
                    papers_referenced.add(papers_ref)
        
        # Calculate overall hallucination score (average)
        avg_hallucination_score = (
            sum(hallucination_scores) / len(hallucination_scores)
            if hallucination_scores
            else 0.0
        )
        
        # Determine if valid
        is_valid = avg_hallucination_score < 0.4  # Relaxed for research context
        
        return {
            "is_valid": is_valid,
            "fact_checks": all_fact_checks,
            "hallucination_score": avg_hallucination_score,
            "citations": list(all_citations),
            "issues": all_issues,
            "papers_referenced": sum(papers_referenced) if papers_referenced else 0
        }


# ============================================================================
# UNIFIED CONTEXT BUILDER
# ============================================================================

class ContextBuilder:
    """Build unified context from all agent results for scientific literature."""
    
    @staticmethod
    def build_unified_context(
        retrieval_results: List[Dict[str, Any]],
        knowledge_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build unified context from retrieval and knowledge results.
        
        Args:
            retrieval_results: Retrieval agent results
            knowledge_results: Knowledge agent results
        
        Returns:
            Unified context dictionary
        """
        logger.info("Building unified context from all results")
        
        # Aggregate retrieval results
        chunks = RetrievalAggregator.aggregate_chunks(retrieval_results)
        graph_facts = RetrievalAggregator.aggregate_graph_facts(retrieval_results)
        sources_set = set(RetrievalAggregator.aggregate_sources(retrieval_results))
        retrieval_metrics = RetrievalAggregator.calculate_aggregate_metrics(retrieval_results)
        
        # Aggregate knowledge results
        knowledge_graph = KnowledgeAggregator.build_knowledge_graph(knowledge_results)
        
        # ✅ FIX: Only add "Knowledge Graph" if there's actual graph data
        has_graph_facts = len(graph_facts) > 0
        has_knowledge_data = (
            knowledge_graph.get("entity_count", 0) > 0 or
            knowledge_graph.get("relationship_count", 0) > 0 or
            knowledge_graph.get("timeline_event_count", 0) > 0
        )

        if has_graph_facts or has_knowledge_data:
            sources_set.add("Knowledge Graph")

        sources = sorted(list(sources_set))

        # Build unified context
        context = {
            # Retrieval data
            "chunks": chunks,
            "graph_facts": graph_facts,
            "sources": sources,
            "retrieval_metrics": retrieval_metrics,
            
            # Knowledge graph data
            "entities": knowledge_graph["entities"],
            "relationships": knowledge_graph["relationships"],
            "timeline": knowledge_graph["timeline"],
            "knowledge_graph": knowledge_graph,
            
            # Statistics
            "stats": {
                "total_chunks": len(chunks),
                "total_facts": len(graph_facts),
                "total_entities": len(knowledge_graph["entities"]),
                "total_relationships": len(knowledge_graph["relationships"]),
                "total_timeline_events": len(knowledge_graph["timeline"]),
                "total_sources": len(sources),
                "total_papers": retrieval_metrics.get("unique_papers", 0),
                "avg_retrieval_score": retrieval_metrics["avg_score"]
            }
        }
        
        logger.info(
            f"Context built: {context['stats']['total_chunks']} chunks, "
            f"{context['stats']['total_papers']} papers, "
            f"{context['stats']['total_entities']} entities, "
            f"{context['stats']['total_relationships']} relationships"
        )
        
        return context


# ============================================================================
# RESULT RANKING AND FILTERING
# ============================================================================

class ResultRanker:
    """Rank and filter aggregated results for scientific papers."""
    
    @staticmethod
    def rank_chunks_by_relevance(
        chunks: List[Dict[str, Any]],
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rank paper chunks by relevance to query.
        
        Args:
            chunks: List of chunks
            query: Query text
            top_k: Number of top chunks to return
        
        Returns:
            Top-k ranked chunks
        """
        # Already sorted by score in aggregation
        return chunks[:top_k]
    
    @staticmethod
    def filter_low_quality_results(
        chunks: List[Dict[str, Any]],
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Filter out low-quality results.
        
        Args:
            chunks: List of chunks
            min_score: Minimum score threshold
        
        Returns:
            Filtered chunks
        """
        filtered = [
            chunk for chunk in chunks
            if chunk.get('score', 0.0) >= min_score
        ]
        
        logger.info(
            f"Filtered {len(chunks)} chunks to {len(filtered)} "
            f"(min_score={min_score})"
        )
        
        return filtered
    
    @staticmethod
    def prioritize_recent_papers(
        results: List[Dict[str, Any]],
        date_field: str = 'publication_date'
    ) -> List[Dict[str, Any]]:
        """
        Prioritize more recent papers.
        
        Args:
            results: List of results with publication dates
            date_field: Field containing date
        
        Returns:
            Sorted results (newest first)
        """
        def get_date(result):
            date_str = result.get(date_field, '')
            if date_str:
                try:
                    return datetime.fromisoformat(date_str)
                except:
                    pass
            return datetime.min
        
        sorted_results = sorted(results, key=get_date, reverse=True)
        
        return sorted_results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def aggregate_all_results(
    retrieval_results: List[Dict[str, Any]],
    knowledge_results: List[Dict[str, Any]],
    synthesis_results: List[Dict[str, Any]],
    validation_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate all agent results into unified structure.
    
    Args:
        retrieval_results: Retrieval agent results
        knowledge_results: Knowledge agent results
        synthesis_results: Synthesis agent results
        validation_results: Validation agent results
    
    Returns:
        Aggregated results for scientific literature
    """
    logger.info("Aggregating all agent results for scientific literature")
    
    # Build unified context
    context = ContextBuilder.build_unified_context(
        retrieval_results=retrieval_results,
        knowledge_results=knowledge_results
    )
    
    # Merge synthesis
    synthesis = SynthesisAggregator.merge_synthesis_results(synthesis_results)
    
    # Merge validation
    validation = ValidationAggregator.merge_validation_reports(validation_results)
    
    # Combine everything
    aggregated = {
        "context": context,
        "synthesis": synthesis,
        "validation": validation,
        
        # Statistics
        "summary": {
            "retrieval_results": len(retrieval_results),
            "knowledge_results": len(knowledge_results),
            "synthesis_results": len(synthesis_results),
            "validation_results": len(validation_results),
            
            "total_chunks": context["stats"]["total_chunks"],
            "total_papers": context["stats"]["total_papers"],
            "total_entities": context["stats"]["total_entities"],
            "total_sources": context["stats"]["total_sources"],
            
            "answer_confidence": synthesis.get("confidence", 0.0),
            "papers_cited": synthesis.get("papers_cited", 0),
            "validation_status": validation.get("is_valid", False),
            "hallucination_score": validation.get("hallucination_score", 0.0),
            "papers_referenced": validation.get("papers_referenced", 0)
        }
    }
    
    logger.info(f"Aggregation complete: {aggregated['summary']}")
    
    return aggregated


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example showing result aggregation for scientific literature."""
    
    # Mock retrieval results
    retrieval_results = [
        {
            "data": {
                "chunks": [
                    {
                        "chunk_id": "1",
                        "content": "Transformer architecture proposed by Vaswani et al.",
                        "score": 0.9,
                        "document_id": "attention_paper",
                        "document_title": "Attention Is All You Need"
                    },
                    {
                        "chunk_id": "2",
                        "content": "BERT uses bidirectional Transformer",
                        "score": 0.85,
                        "document_id": "bert_paper",
                        "document_title": "BERT: Pre-training of Deep Bidirectional Transformers"
                    }
                ],
                "graph_facts": [
                    {"uuid": "fact1", "fact": "Transformer proposed at NeurIPS 2017"}
                ],
                "sources": ["attention_paper.pdf"],
                "total_results": 2
            }
        }
    ]
    
    # Mock knowledge results
    knowledge_results = [
        {
            "data": {
                "entities": [
                    {"name": "Transformer", "type": "METHOD"},
                    {"name": "Vaswani", "type": "AUTHOR"}
                ],
                "relationships": [
                    {"source": "Vaswani", "target": "Transformer", "type": "PROPOSED"}
                ],
                "timeline": [
                    {"date": "2017-06", "event": "Transformer architecture proposed"}
                ]
            }
        }
    ]
    
    print("="*70)
    print("RESULT AGGREGATION FOR SCIENTIFIC LITERATURE")
    print("="*70)
    
    # Aggregate retrieval results
    print("\n1. Aggregating paper retrieval results...")
    chunks = RetrievalAggregator.aggregate_chunks(retrieval_results)
    print(f"   Total chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"   - {chunk['chunk_id']}: {chunk['content'][:50]}... (score: {chunk['score']})")
    
    # Aggregate knowledge results
    print("\n2. Aggregating academic knowledge results...")
    kg = KnowledgeAggregator.build_knowledge_graph(knowledge_results)
    print(f"   Entities: {kg['entity_count']}")
    print(f"   Relationships: {kg['relationship_count']}")
    print(f"   Timeline events: {kg['timeline_event_count']}")
    
    # Build unified context
    print("\n3. Building unified context for research...")
    context = ContextBuilder.build_unified_context(retrieval_results, knowledge_results)
    print(f"   Context stats: {context['stats']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    example_usage()
