"""

SCINOBO RESEARCH ARTIFACT ANALYSIS API SERVER

"""

import os
import traceback
import logging
import importlib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple
from enum import Enum
from fastapi.middleware.cors import CORSMiddleware

from raa.server.logging_setup import setup_root_logger
from raa.pipeline.inference import extract_research_artifacts_text_list_fast_mode, extract_research_artifacts_text_list

BASE_PATH = importlib.resources.files(__package__.split(".")[0])
DATA_PATH = os.path.join(BASE_PATH, 'data')

app = FastAPI()

# init the logger
setup_root_logger()
logger = logging.getLogger(__name__)
logger.info("RAA Api initialized")

class GazetteerChoices(str, Enum):
    hybrid = "hybrid"
    synthetic = "synthetic"

class RAAInferRequest(BaseModel):
    text_list: List[List[str]]
    fast_mode: bool = False
    perform_deduplication: bool = False
    insert_fast_mode_gazetteers: bool = False
    dataset_gazetteers: GazetteerChoices | None = None
    split_sentences: bool = False

class RAAInferResponse(BaseModel):
    text_list: List[List[str]]
    research_artifacts: Dict[str, Any]

class RAAFastInferResponse(BaseModel):
    text_list: List[List[str]]
    research_artifacts: List[Dict[str, Any]]

class ErrorResponse(BaseModel):
    success: int
    message: str

app = FastAPI()

# handle CORS -- at a later stage we can restrict the origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/infer_text_list', response_model=RAAInferResponse|RAAFastInferResponse, responses={400: {"model": ErrorResponse}})
def infer_citances(request_data: RAAInferRequest):
    try:
        logger.debug("JSON received...")
        logger.debug(request_data.json())
        
        text_list = request_data.text_list
        fast_mode = request_data.fast_mode
        perform_deduplication = request_data.perform_deduplication
        insert_fast_mode_gazetteers = request_data.insert_fast_mode_gazetteers
        dataset_gazetteers = request_data.dataset_gazetteers
        split_sentences = request_data.split_sentences

        if fast_mode:
            output = extract_research_artifacts_text_list_fast_mode(text_list, split_sentences=split_sentences, dataset_gazetteers=dataset_gazetteers)
            return RAAFastInferResponse(text_list=output['text_list'], research_artifacts=output['research_artifacts'])
        else:
            output = extract_research_artifacts_text_list(text_list, split_sentences=split_sentences, perform_deduplication=perform_deduplication, insert_fast_mode_gazetteers=insert_fast_mode_gazetteers, dataset_gazetteers=dataset_gazetteers)
            
            return RAAInferResponse(text_list=output['text_list'], research_artifacts=output['research_artifacts'])

    except Exception as e:
        logger.error(str(e))
        return HTTPException(status_code=400, detail={"success": 0, "message": f"{str(e)}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT", 8000))
