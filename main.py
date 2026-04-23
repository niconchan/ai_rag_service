from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
from pathlib import Path
import tempfile
import logging

from ingestion import IngestionPipeline
from retriever import RAGRetriever, RetrievalResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Service with Ray and Milvus")

_ingestion_pipeline: Optional[IngestionPipeline] = None
_retriever: Optional[RAGRetriever] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    results: List[dict]


class IngestRequest(BaseModel):
    folder_path: str
    rebuild_collection: bool = False


@app.on_event("startup")
async def startup_event():
    global _ingestion_pipeline, _retriever
    _ingestion_pipeline = IngestionPipeline()
    _retriever = RAGRetriever()
    logger.info("RAG Service started")


@app.on_event("shutdown")
async def shutdown_event():
    global _ingestion_pipeline, _retriever
    if _ingestion_pipeline:
        _ingestion_pipeline.close()
    if _retriever:
        _retriever.close()
    logger.info("RAG Service stopped")


@app.post("/ingest/pdf", response_model=JSONResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    rebuild_collection: bool = Form(False)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        _ingestion_pipeline.ingest_pdf(tmp_path, rebuild_collection=rebuild_collection)

        return JSONResponse(content={"message": f"Successfully ingested {file.filename}"})

    except Exception as e:
        logger.error(f"Error ingesting PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ingest/folder", response_model=JSONResponse)
async def ingest_folder(request: IngestRequest):
    folder_path = Path(request.folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid folder path")

    try:
        _ingestion_pipeline.ingest_folder(
            str(folder_path),
            rebuild_collection=request.rebuild_collection
        )
        return JSONResponse(content={"message": f"Successfully ingested folder: {request.folder_path}"})

    except Exception as e:
        logger.error(f"Error ingesting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if request.top_k < 1 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")

    try:
        results = _retriever.retrieve(request.query, top_k=request.top_k)

        response_data = [
            {
                "content": r.content,
                "distance": r.distance,
                "source": r.metadata.get("source"),
                "page": r.metadata.get("page")
            }
            for r in results
        ]

        return QueryResponse(results=response_data)

    except Exception as e:
        logger.error(f"Error querying: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_with_context", response_model=JSONResponse)
async def query_with_context(request: QueryRequest):
    if request.top_k < 1 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")

    try:
        context = _retriever.retrieve_with_context(
            request.query,
            top_k=request.top_k,
            max_context_length=2000
        )

        return JSONResponse(content={
            "query": request.query,
            "context": context
        })

    except Exception as e:
        logger.error(f"Error querying with context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
