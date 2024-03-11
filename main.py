from transformers import AutoModelForCausalLM, AutoTokenizer , BitsAndBytesConfig
import transformers
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from threading import Thread
from transformers import TextIteratorStreamer
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import uvicorn
import time
import locale
import os


model_id = None

def load_model_and_tokenizer(model_id):
    # Define the quantization configuration for the model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("Loading model...")

    # Load the model using the model ID and quantization configuration
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, trust_remote_code=True, device_map='auto')

    # Load the tokenizer associated with the model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Model & tokenizer loaded from ", model_id)
    return model, tokenizer


def setup_text_generation_pipeline(model, tokenizer):
    # Initialize a TextIteratorStreamer object for streaming text generation
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

    # Create a text generation pipeline using the Hugging Face transformers library
    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.3,
        repetition_penalty=1.1,
        max_new_tokens=1000,
        do_sample=True,
        device_map='auto',
        streamer=streamer  # Use the streamer for streaming text generation
    )
    return text_generation_pipeline, streamer


def initialize_application():
    model_id = os.getenv('MODEL_ID', 'mistralai/Mistral-7B-Instruct-v0.2')  # Default model ID if not set in env
    print("Model selected: ", model_id)
    locale.getpreferredencoding = lambda: "UTF-8"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    model, tokenizer = load_model_and_tokenizer(model_id)
    text_generation_pipeline, streamer = setup_text_generation_pipeline(model, tokenizer)
    
    # Create LLM & chain
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    # Define prompt template
    prompt_template = """
    ### [INST]
    Instruction: I will ask you a QUESTION and give you a CONTEXT and you will respond with an answer easily understandable.

    ### CONTEXT:
    {context}

    ### QUESTION:
    {question}

    [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    llm_chain = prompt | llm

    print("Created LLM and chain")

    # Your additional setup code here (e.g., FastAPI app initialization)
    app = FastAPI()

    # Variables for time measurements
    start_time = 0
    first_token_time = 0
    token_times = []

    # Invoke the LLM chain using the input text
    def invoke_llm_chain(input_text):
        llm_chain.invoke(input_text)


    # Function to calculate metrics
    def calculate_metrics(start_time, first_token_time, token_times, model_output):
        end_time = time.time()
        end_to_end_latency = end_time - start_time
        ttft = first_token_time - start_time
        itl = sum(x - y for x, y in zip(token_times[1:], token_times[:-1])) / (len(token_times) - 1)

        throughput = len(model_output) / end_to_end_latency
        return {
            "End-to-end Latency": end_to_end_latency,
            "Time To First Token (TTFT)": ttft,
            "Inter-token latency (ITL)": itl,
            "Throughput": throughput
        }

    # Generate output text using the streamer
    def generate_output(streamer):
        global start_time, first_token_time, token_times, model_output
        model_output = ""
        start_time = time.time()

        for i, new_text in enumerate(streamer):
            model_output += new_text

            # Measure time for the first token
            if i == 0:
                first_token_time = time.time()

            # Measure time for each token
            token_times.append(time.time())
            yield new_text

        metrics = calculate_metrics(start_time, first_token_time, token_times, model_output)
        print("Metrics:", metrics)
        return metrics
    
    @app.get("/")
    async def root():
        return {"message": "Hello, World!"}

    @app.post("/inference")
    async def inference(input_text: dict, background_tasks: BackgroundTasks):
        # Start a separate thread to run the LLM chain asynchronously
        thread = Thread(target=invoke_llm_chain, args=[input_text])
        thread.start()

        # Add the generate_output function to the background tasks with the streamer
        background_tasks.add_task(generate_output, streamer)

        return StreamingResponse(generate_output(streamer))

    return app

app = initialize_application()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-m', '--model', required=True)
#     args = parser.parse_args()
#     print("Model selected: ", args.model)
#     model_id='mistralai/Mistral-7B-Instruct-v0.2'

#     app = initialize_application(model_id)
    # Run server
    # uvicorn.run(app)
    # python main.py --model mistralai/Mistral-7B-Instruct-v0.2









# #########
# # Setup
# #########

# locale.getpreferredencoding = lambda: "UTF-8"

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Device: ", device)

# # Define the model ID for the desired model
# # model_id = "mistralai/Mistral-7B-Instruct-v0.2"


# # Define the quantization configuration for the model
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# print("Loading model...")

# # Load the model using the model ID and quantization configuration
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, trust_remote_code=True, device_map='auto')

# # Load the tokenizer associated with the model
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# print("Model & tokenizer loaded from ", model_id)

# # Initialize a TextIteratorStreamer object for streaming text generation
# streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

# # Create a text generation pipeline using the Hugging Face transformers library
# text_generation_pipeline = transformers.pipeline(
#     model=model,
#     tokenizer=tokenizer,
#     task="text-generation",
#     temperature=0.3,
#     repetition_penalty=1.1,
#     max_new_tokens=1000,
#     do_sample=True,
#     device_map='auto',
#     streamer=streamer  # Use the streamer for streaming text generation
# )

# # Define prompt template
# prompt_template = """
# ### [INST]
# Instruction: I will ask you a QUESTION and give you a CONTEXT and you will respond with an answer easily understandable.

# ### CONTEXT:
# {context}

# ### QUESTION:
# {question}

# [/INST]
#  """

# # Create LLM & chain
# llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=prompt_template,
# )

# llm_chain = prompt | llm

# print("Created LLM and chain")



# #########
# # API
# #########

# from fastapi import FastAPI, Request, BackgroundTasks
# from fastapi.responses import StreamingResponse
# from threading import Thread

# app = FastAPI()

# # Variables for time measurements
# start_time = 0
# first_token_time = 0
# token_times = []

# # Invoke the LLM chain using the input text
# def invoke_llm_chain(input_text):
#     llm_chain.invoke(input_text)


# # Function to calculate metrics
# def calculate_metrics(start_time, first_token_time, token_times, model_output):
#     end_time = time.time()
#     end_to_end_latency = end_time - start_time
#     ttft = first_token_time - start_time
#     itl = sum(x - y for x, y in zip(token_times[1:], token_times[:-1])) / (len(token_times) - 1)

#     throughput = len(model_output) / end_to_end_latency
#     return {
#         "End-to-end Latency": end_to_end_latency,
#         "Time To First Token (TTFT)": ttft,
#         "Inter-token latency (ITL)": itl,
#         "Throughput": throughput
#     }

# # Generate output text using the streamer
# def generate_output(streamer):
#     global start_time, first_token_time, token_times, model_output
#     model_output = ""
#     start_time = time.time()

#     for i, new_text in enumerate(streamer):
#         model_output += new_text

#         # Measure time for the first token
#         if i == 0:
#             first_token_time = time.time()

#         # Measure time for each token
#         token_times.append(time.time())
#         yield new_text

#     metrics = calculate_metrics(start_time, first_token_time, token_times, model_output)
#     print("Metrics:", metrics)
#     return metrics


# @app.get("/")
# async def root():
#     return {"message": "Hello, World!"}


# @app.post("/inference")
# async def inference(input_text: dict, background_tasks: BackgroundTasks):
#     # Start a separate thread to run the LLM chain asynchronously
#     thread = Thread(target=invoke_llm_chain, args=[input_text])
#     thread.start()

#     # Add the generate_output function to the background tasks with the streamer
#     background_tasks.add_task(generate_output, streamer)

#     return StreamingResponse(generate_output(streamer))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-m', '--model')
#     args = parser.parse_args()
#     print("Model selected: ", args.model)
#     model_id = args.model

#     #run server
#     uvicorn.run(app)
#     # python main.py --model mistralai/Mistral-7B-Instruct-v0.2