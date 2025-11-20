"""
Document chunker optimized for scientific papers.
Simple, reliable chunking that preserves academic structure.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Tipo de chunk no paper."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    CONTENT = "content"


@dataclass
class ChunkingConfig:
    """Configuração de chunking."""
    chunk_size: int = 800  # Reduzido para papers (melhor para embeddings)
    chunk_overlap: int = 100  # Pequeno overlap para continuidade
    max_chunk_size: int = 1500
    min_chunk_size: int = 50
    preserve_academic_structure: bool = True  # Respeita seções


@dataclass
class DocumentChunk:
    """Representa um chunk de documento."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Calcula token count se não informado."""
        if self.token_count is None:
            # ~4 chars per token (OpenAI standard)
            self.token_count = len(self.content) // 4


class AcademicPaperChunker:
    """Chunker otimizado para papers acadêmicos."""
    
    def __init__(self, config: ChunkingConfig):
        """Inicializa chunker."""
        self.config = config
        
        # Padrões para identificar seções de papers
        self.section_patterns = {
            ChunkType.ABSTRACT: r'\b(?:abstract|summary)\b',
            ChunkType.INTRODUCTION: r'\b(?:introduction|background)\b',
            ChunkType.METHODOLOGY: r'\b(?:method|approach|algorithm|proposed|proposed method)\b',
            ChunkType.EXPERIMENTS: r'\b(?:experiment|evaluation|result|benchmark)\b',
            ChunkType.RESULTS: r'\b(?:results?|finding|outcome)\b',
            ChunkType.CONCLUSION: r'\b(?:conclusion|discussion|future work)\b',
            ChunkType.REFERENCES: r'\b(?:reference|bibliography|citation)\b'
        }
    
    def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunifica documento respeitando estrutura acadêmica.
        
        Args:
            content: Conteúdo do documento
            title: Título do documento
            source: Fonte do documento
            metadata: Metadados adicionais
        
        Returns:
            Lista de chunks do documento
        """
        if not content.strip():
            return []
        
        logger.info(f"Chunking document: {title} ({len(content)} chars)")
        
        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "academic_structure_aware",
            **(metadata or {})
        }
        
        # Remove/normaliza whitespace
        content = self._normalize_content(content)
        
        # Identifica seções acadêmicas se configurado
        if self.config.preserve_academic_structure:
            chunks = self._chunk_by_academic_structure(content, base_metadata)
        else:
            chunks = self._chunk_by_paragraphs(content, base_metadata)
        
        # Valida chunks
        chunks = [c for c in chunks if len(c.content.strip()) >= self.config.min_chunk_size]
        
        # Atualiza índices e metadados
        for i, chunk in enumerate(chunks):
            chunk.index = i
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["chunk_type"] = self._detect_chunk_type(chunk.content)
        
        logger.info(f"Created {len(chunks)} chunks (avg {len(content)//len(chunks) if chunks else 0} chars)")
        
        return chunks
    
    def _normalize_content(self, content: str) -> str:
        """Normaliza whitespace e caracteres especiais."""
        # Remove múltiplos espaços
        content = re.sub(r' {2,}', ' ', content)
        # Remove múltiplas newlines (máximo 2)
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()
    
    def _chunk_by_academic_structure(
        self,
        content: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunifica respeitando seções de paper acadêmico.
        
        Estratégia:
        1. Identifica seções principais
        2. Chunifica cada seção respeitando chunk_size
        3. Mantém pequeno overlap entre chunks
        """
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        # Split por seções (headers)
        # Busca padrões de header: "# Seção", "## Subseção", "### Subsubseção"
        section_pattern = r'^(#{1,3})\s+(.+?)$'
        lines = content.split('\n')
        
        current_section = ""
        current_section_title = ""
        
        for i, line in enumerate(lines):
            match = re.match(section_pattern, line, re.MULTILINE)
            
            if match and current_section:
                # Encontrou nova seção, processa a anterior
                chunks.extend(
                    self._chunk_section(
                        current_section,
                        current_section_title,
                        base_metadata,
                        chunk_index
                    )
                )
                chunk_index = len(chunks)
                current_section = line
                current_section_title = match.group(2)
            else:
                current_section += "\n" + line if current_section else line
        
        # Processa última seção
        if current_section:
            chunks.extend(
                self._chunk_section(
                    current_section,
                    current_section_title,
                    base_metadata,
                    chunk_index
                )
            )
        
        # Se não encontrou seções, usa chunking por parágrafo
        if not chunks:
            chunks = self._chunk_by_paragraphs(content, base_metadata)
        
        return chunks
    
    def _chunk_section(
        self,
        section_content: str,
        section_title: str,
        base_metadata: Dict[str, Any],
        start_chunk_index: int
    ) -> List[DocumentChunk]:
        """Chunifica uma seção do paper."""
        chunks = []
        
        # Split por parágrafos
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', section_content) if p.strip()]
        
        if not paragraphs:
            return chunks
        
        current_chunk = ""
        chunk_index = start_chunk_index
        start_pos = 0
        
        for para in paragraphs:
            # Tenta adicionar parágrafo ao chunk atual
            potential_chunk = (current_chunk + "\n\n" + para) if current_chunk else para
            
            if len(potential_chunk) <= self.config.chunk_size:
                # Cabe no chunk, adiciona
                current_chunk = potential_chunk
            else:
                # Não cabe, salva chunk atual e começa novo
                if current_chunk:
                    chunk_obj = DocumentChunk(
                        content=current_chunk.strip(),
                        index=chunk_index,
                        start_char=start_pos,
                        end_char=start_pos + len(current_chunk),
                        metadata={
                            **base_metadata,
                            "section": section_title,
                            "section_type": self._detect_section_type(section_title)
                        }
                    )
                    chunks.append(chunk_obj)
                    chunk_index += 1
                    
                    # Overlap para continuidade
                    overlap_len = min(self.config.chunk_overlap, len(current_chunk) // 2)
                    start_pos += len(current_chunk) - overlap_len
                    
                    # Começa novo chunk com overlap do anterior
                    current_chunk = current_chunk[-overlap_len:] + "\n\n" + para if overlap_len > 0 else para
                else:
                    # Parágrafo sozinho é maior que chunk_size, divide-o
                    sub_chunks = self._split_long_paragraph(
                        para,
                        section_title,
                        base_metadata,
                        chunk_index
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    current_chunk = ""
                    start_pos += len(para)
        
        # Salva último chunk
        if current_chunk:
            chunk_obj = DocumentChunk(
                content=current_chunk.strip(),
                index=chunk_index,
                start_char=start_pos,
                end_char=start_pos + len(current_chunk),
                metadata={
                    **base_metadata,
                    "section": section_title,
                    "section_type": self._detect_section_type(section_title)
                }
            )
            chunks.append(chunk_obj)
        
        return chunks
    
    def _chunk_by_paragraphs(
        self,
        content: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunifica por parágrafos simples."""
        chunks = []
        
        # Split por parágrafos
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        current_pos = 0
        
        for para in paragraphs:
            potential = (current_chunk + "\n\n" + para) if current_chunk else para
            
            if len(potential) <= self.config.chunk_size:
                current_chunk = potential
            else:
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        index=chunk_index,
                        start_char=current_pos,
                        end_char=current_pos + len(current_chunk),
                        metadata=base_metadata.copy()
                    ))
                    chunk_index += 1
                    
                    # Overlap
                    overlap = min(self.config.chunk_overlap, len(current_chunk) // 2)
                    current_pos += len(current_chunk) - overlap
                    current_chunk = current_chunk[-overlap:] + "\n\n" + para if overlap > 0 else para
                else:
                    # Para muito longo
                    sub_chunks = self._split_long_paragraph(para, "", base_metadata, chunk_index)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    current_pos += len(para)
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                index=chunk_index,
                start_char=current_pos,
                end_char=current_pos + len(current_chunk),
                metadata=base_metadata.copy()
            ))
        
        return chunks
    
    def _split_long_paragraph(
        self,
        para: str,
        section: str,
        metadata: Dict[str, Any],
        start_index: int
    ) -> List[DocumentChunk]:
        """Divide parágrafo longo em chunks menores."""
        chunks = []
        
        sentences = re.split(r'(?<=[.!?])\s+', para)
        current = ""
        chunk_idx = start_index
        pos = 0
        
        for sent in sentences:
            if not sent.strip():
                continue
            
            potential = (current + " " + sent) if current else sent
            
            if len(potential) <= self.config.max_chunk_size:
                current = potential
            else:
                if current:
                    chunks.append(DocumentChunk(
                        content=current.strip(),
                        index=chunk_idx,
                        start_char=pos,
                        end_char=pos + len(current),
                        metadata={
                            **metadata,
                            "section": section,
                            "from_long_paragraph": True
                        }
                    ))
                    chunk_idx += 1
                    pos += len(current)
                    current = sent
                else:
                    # Sentença sozinha é muito longa, trunca
                    chunks.append(DocumentChunk(
                        content=sent[:self.config.max_chunk_size],
                        index=chunk_idx,
                        start_char=pos,
                        end_char=pos + len(sent[:self.config.max_chunk_size]),
                        metadata={
                            **metadata,
                            "section": section,
                            "truncated": True
                        }
                    ))
                    chunk_idx += 1
                    pos += len(sent)
                    current = ""
        
        if current:
            chunks.append(DocumentChunk(
                content=current.strip(),
                index=chunk_idx,
                start_char=pos,
                end_char=pos + len(current),
                metadata={
                    **metadata,
                    "section": section,
                    "from_long_paragraph": True
                }
            ))
        
        return chunks
    
    def _detect_chunk_type(self, content: str) -> str:
        """Detecta tipo de chunk."""
        content_lower = content.lower()[:500]
        
        for chunk_type, pattern in self.section_patterns.items():
            if re.search(pattern, content_lower, re.IGNORECASE):
                return chunk_type.value
        
        return ChunkType.CONTENT.value
    
    def _detect_section_type(self, section_title: str) -> str:
        """Detecta tipo de seção."""
        title_lower = section_title.lower()
        
        for chunk_type, pattern in self.section_patterns.items():
            if re.search(pattern, title_lower, re.IGNORECASE):
                return chunk_type.value
        
        return ChunkType.CONTENT.value


# Factory
def create_chunker(config: ChunkingConfig = None) -> AcademicPaperChunker:
    """Factory para criar chunker."""
    if config is None:
        config = ChunkingConfig()
    return AcademicPaperChunker(config)
