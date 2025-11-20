"""
PDF to Markdown converter using Docling (high-quality extraction).
Skips PDFs that already have converted markdown.
"""
import os
from pathlib import Path
from typing import List, Dict
import logging
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


class PDFConverter:
    """Convert PDFs to Markdown using Docling."""
    
    def __init__(self, input_dir: str = "pdfs/", output_dir: str = "markdowns/"):
        """
        Initialize converter.
        
        Args:
            input_dir: Directory containing PDFs
            output_dir: Directory for markdown output
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize Docling converter
        self.converter = DocumentConverter()
    
    def _markdown_exists(self, pdf_path: Path) -> bool:
        """Check if markdown already exists for PDF."""
        markdown_file = self.output_dir / f"{pdf_path.stem}.md"
        exists = markdown_file.exists()
        
        if exists:
            logger.info(f"⏭️  Skipping (already exists): {pdf_path.name}")
        
        return exists
    
    def convert_single(self, pdf_path: str) -> Dict[str, str]:
        """
        Convert single PDF to markdown.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with metadata and markdown path, or None if skipped
        """
        pdf_file = Path(pdf_path)
        
        # Skip if markdown already exists
        if self._markdown_exists(pdf_file):
            return None
        
        try:
            logger.info(f"Converting: {pdf_file.name}")
            
            # Convert with Docling
            result = self.converter.convert(str(pdf_file))
            
            # Generate output path
            output_file = self.output_dir / f"{pdf_file.stem}.md"
            
            # Export to markdown
            markdown_content = result.document.export_to_markdown()
            output_file.write_text(markdown_content, encoding="utf-8")
            
            logger.info(f"✓ Saved: {output_file.name}")
            
            # Extract metadata
            metadata = {
                "title": result.document.name or pdf_file.stem,
                "pages": len(result.pages) if hasattr(result, 'pages') else None,
                "markdown_path": str(output_file),
                "pdf_path": str(pdf_file)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to convert {pdf_file.name}: {e}")
            return None
    
    def convert_batch(self) -> Dict[str, any]:
        """
        Convert all PDFs in input directory, skipping already converted ones.
        
        Returns:
            Dict with conversion stats and metadata
        """
        pdf_files = sorted(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDFs found in {self.input_dir}")
            return {"total": 0, "converted": 0, "skipped": 0, "failed": 0, "results": []}
        
        logger.info(f"Found {len(pdf_files)} PDFs to process")
        
        results = []
        skipped = 0
        failed = 0
        
        for pdf_file in pdf_files:
            metadata = self.convert_single(str(pdf_file))
            
            if metadata is None:
                # Check if it was skipped or failed
                if self._markdown_exists(pdf_file):
                    skipped += 1
                else:
                    failed += 1
            else:
                results.append(metadata)
        
        converted = len(results)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Conversion Summary")
        logger.info(f"{'='*50}")
        logger.info(f"Total PDFs:    {len(pdf_files)}")
        logger.info(f"Converted:     {converted}")
        logger.info(f"Skipped:       {skipped}")
        logger.info(f"Failed:        {failed}")
        logger.info(f"{'='*50}\n")
        
        return {
            "total": len(pdf_files),
            "converted": converted,
            "skipped": skipped,
            "failed": failed,
            "results": results
        }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown (skip existing)")
    parser.add_argument(
        "--input-dir",
        default="pdfs/",
        help="Directory containing PDFs"
    )
    parser.add_argument(
        "--output-dir",
        default="markdowns/",
        help="Directory for markdown output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Convert
    converter = PDFConverter(args.input_dir, args.output_dir)
    stats = converter.convert_batch()


if __name__ == "__main__":
    main()
