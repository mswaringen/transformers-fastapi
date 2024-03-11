# Transformers FastAPI

Run Huggingface Transformer models via FastAPI, locally or remote. This is a proof of concept and not production-ready code.

## [RunPod](https://www.runpod.io/) Instructions (remote)
#### Create Pod

- Manage -> Pods -> +GPU Pod
- From the Secure Cloud drop down select Community Cloud
- Scroll down to previous generation and choose something cheap
- Customize deployment
- Expose HTTP Ports -> add 8000
- Add Environment Variable `MODEL_ID` (if not specified default is MODEL_ID='mistralai/Mistral-7B-Instruct-v0.2’) to download model from Huggingface
- Click Continue and then Deploy

#### Install & Run Inference Server
Once deployed click drop down on your pod then click Connect -> Start Web Terminal -> connect to web terminal (wait for system to setup)
enter the following command:

`cd workspace && \
git clone https://github.com/mswaringen/transformers-fastapi.git && \
cd transformers-fastapi && \
pip install -r requirements.txt && \
uvicorn main:app --host 0.0.0.0 --port 8000`



## Run Server Locally

- Set up local venv/pipenv/conda etc
- Modify .env
    `MODEL_ID='microsoft/phi-2'
    QUANTIZE_MODEL=FALSE`
- Run the following command

`git clone https://github.com/mswaringen/transformers-fastapi.git && \
cd transformers-fastapi && \
pip install -r requirements.txt && \
uvicorn main:app --host 0.0.0.0 --port 8000`


## Test the API
- Modify test_api.py to set URL as needed
- Run `python3 test_api.py`