""" 

This test script will be used to test the API endpoints for the SciNoBo-RAA.

"""
# ------------------------------------------------------------ #
import sys
sys.path.append("./src") # since it is not installed yet, we need to add the path to the module 
# -- this is for when cloning the repo
# ------------------------------------------------------------ #
from fastapi.testclient import TestClient
from raa.server.api import app

client = TestClient(app)

def test_valid_request():
    request_data = {
        "text_list": [
                ["this dataset is called BIOMRC"],
                ["BIOREAD is a dataset used to train the new MRC models"]
            ],
        "fast_mode": False,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": None,
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 200
    assert "research_artifacts" in response.json()

def test_valid_request_deduplication():
    request_data = {
        "text_list": [
                ["this dataset is called BIOMRC"],
                ["BIOREAD is a dataset used to train the new MRC models"]
            ],
        "fast_mode": False,
        "perform_deduplication": True,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": None,
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 200
    assert "research_artifacts" in response.json()

def test_valid_request_insert_fast_mode_gazetteers():
    request_data = {
        "text_list": [
                ["this dataset is called BIOMRC"],
                ["BIOREAD is a dataset used to train the new MRC models"]
            ],
        "fast_mode": False,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": True,
        "dataset_gazetteers": None,
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 200
    assert "research_artifacts" in response.json()

def test_valid_request_dataset_gazetteers():
    request_data = {
        "text_list": [
                ["this dataset is called BIOMRC"],
                ["BIOREAD is a dataset used to train the new MRC models"]
            ],
        "fast_mode": False,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": "hybrid",
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 200
    assert "research_artifacts" in response.json()

    request_data = {
        "text_list": [
                ["this dataset is called BIOMRC"],
                ["BIOREAD is a dataset used to train the new MRC models"]
            ],
        "fast_mode": False,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": "synthetic",
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 200
    assert "research_artifacts" in response.json()

def test_valid_request_fast_mode():
    request_data = {
        "text_list": [
                ["this dataset is called BIOMRC"],
                ["BIOREAD is a dataset used to train the new MRC models"]
            ],
        "fast_mode": True,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": None,
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 200
    assert "research_artifacts" in response.json()

def test_valid_request_split_sentences():
    request_data = {
        "text_list": [
                ["this dataset is called BIOMRC"],
                ["BIOREAD is a dataset used to train the new MRC models"]
            ],
        "fast_mode": False,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": None,
        "split_sentences": True
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 200
    assert "research_artifacts" in response.json()


def test_invalid_request():
    request_data = {
        "text_list": "invalid_value",
        "fast_mode": False,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": None,
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()

    request_data = {
        "text_list": [["Lorem ipsum dolor sit amet."]],
        "fast_mode": "invalid_value",
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": None,
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()

    request_data = {
        "text_list": [["Lorem ipsum dolor sit amet."]],
        "fast_mode": False,
        "perform_deduplication": "invalid_value",
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": None,
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()

    request_data = {
        "text_list": [["Lorem ipsum dolor sit amet."]],
        "fast_mode": False,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": "invalid_value",
        "dataset_gazetteers": None,
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()

    request_data = {
        "text_list": [["Lorem ipsum dolor sit amet."]],
        "fast_mode": False,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": "invalid_value",
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()

    request_data = {
        "text_list": [["Lorem ipsum dolor sit amet."]],
        "fast_mode": False,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": None,
        "split_sentences": "invalid_value"
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_empty_request():
    request_data = {}
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_empty_list_request():
    request_data = {
        "text_list": [],
        "fast_mode": False,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": None,
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 200
    assert "research_artifacts" in response.json()

def test_missing_values():
    request_data = {
        "text_list": None,
        "fast_mode": False,
        "perform_deduplication": False,
        "insert_fast_mode_gazetteers": False,
        "dataset_gazetteers": None,
        "split_sentences": False
    }
    response = client.post("/infer_text_list", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()
