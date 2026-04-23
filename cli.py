import argparse
import sys
import logging

from ingestion import IngestionPipeline
from retriever import RAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAG CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into Milvus")
    ingest_parser.add_argument("--folder", required=True, help="Folder containing PDFs")
    ingest_parser.add_argument("--rebuild", action="store_true", help="Rebuild collection")

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--query", required=True, help="Query text")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if args.command == "ingest":
        pipeline = IngestionPipeline()
        pipeline.ingest_folder(args.folder, rebuild_collection=args.rebuild)
        pipeline.close()
        logger.info("Ingestion completed")

    elif args.command == "query":
        retriever = RAGRetriever()
        results = retriever.retrieve(args.query, top_k=args.top_k)
        retriever.close()

        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} (distance: {result.distance:.4f}) ---")
            print(f"Source: {result.metadata['source']}, Page: {result.metadata['page']}")
            print(f"Content: {result.content[:200]}...")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
