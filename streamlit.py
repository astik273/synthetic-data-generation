import os
import copy
import tqdm
import tiktoken
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from anthropic import Anthropic, BadRequestError
from langfuse.decorators import observe, langfuse_context
from concurrent.futures import ThreadPoolExecutor, as_completed


EMB_DIM = 1536
MAX_WORKERS = 16
TARGET_DIR = os.environ["TARGET_DIR"]

load_dotenv(dotenv_path=os.environ["ENV_PATH"])

anthropic_client = Anthropic()


def truncate_text(text, model, max_tokens):
    encoding = tiktoken.encoding_for_model(model)

    tokens = encoding.encode(text)
    token_count = len(tokens)
    print(f"Token count: {token_count}")
    
    if token_count > max_tokens:
        tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(tokens)
        return truncated_text
    return text


@observe(as_type="generation")
def _anthropic_completion(**kwargs):
    kwargs_clone = kwargs.copy()
    messages_input = kwargs_clone.pop('messages', None)
    model = kwargs_clone.pop('model', None)
    stream = kwargs.get("stream", False)

    try:
        response = anthropic_client.messages.create(**kwargs)
    except BadRequestError:
        # Try up to 3 times, truncating the message content with each attempt
        for retry in range(3):
            try:
                # Check if we have messages to truncate
                if len(kwargs['messages']) > 0 and 'content' in kwargs['messages'][0]:
                    # Calculate truncation based on the user's formula
                    content = kwargs['messages'][0]['content']
                    mes_len = len(content)
                    now_len = int(mes_len * 0.85)  # Keep 15% of the message
                    
                    # Truncate the message content
                    kwargs['messages'][0]['content'] = content[now_len:]
                    
                    # Try the API call again with the truncated content
                    response = anthropic_client.messages.create(**kwargs)
                    break  # Break the loop if successful
                else:
                    # If there are no messages to truncate, just try with the original parameters
                    response = anthropic_client.messages.create(**kwargs)
                    break
            except BadRequestError:
                # If we've exhausted all retries, use the original fallback method
                if retry == 2:
                    if len(kwargs['messages']) > 25:
                        kwargs['messages'] = kwargs['messages'][10:]
                    response = anthropic_client.messages.create(**kwargs)

    # If streaming is enabled, accumulate the content token-by-token.
    input_tokens, output_tokens = 0, 0
    if stream:
        complete_content = ""
        for idx, chunk in enumerate(response):
            if hasattr(chunk, "delta"):
                delta = chunk.delta
                if hasattr(delta, "text"):
                    complete_content += delta.text

            if idx == 0 and hasattr(chunk, "message"):
                if hasattr(chunk.message, "usage"):
                    input_tokens = (
                        chunk.message.usage.cache_creation_input_tokens
                        + chunk.message.usage.cache_read_input_tokens
                        + chunk.message.usage.input_tokens
                    )
            elif hasattr(chunk, "delta") and hasattr(chunk, "usage"):
                if hasattr(chunk.delta, "stop_reason") and hasattr(chunk.usage, "output_tokens"):
                    if chunk.delta.stop_reason == "end_turn":
                        output_tokens = chunk.usage.output_tokens
            
        content_result = complete_content
    else:
        # Non-streaming behavior (using existing logic)
        if "thinking" in kwargs_clone:
            content_result = response.content[1].text
        else:
            content_result = response.content[0].text

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

    langfuse_context.update_current_observation(
        input=messages_input,
        model=model,
        metadata=kwargs_clone,
        usage_details={
            "input": input_tokens,
            "output": output_tokens
        }
    )

    return content_result


def call_models(messages, provider, model, max_tokens=2048, top_p=0.9, temperature=1.0, stream=False, **extra_kwargs):
    if provider == "fireworks":
        client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ["FIREWORKS_API_KEY"],
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            stream=stream
        )
        if stream:
            complete_output = ""
            for chunk in response:
                token = chunk.choices
                if len(token) > 0:
                    token = token[0].delta.content
                else:
                    token = ""
                token = "" if token is None else token
                complete_output += token

            return complete_output
        else:
            return response.choices[0].message.content

    elif provider == "openai":
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
        if model.startswith("text-embedding"):
            # Determine the input text. If messages is a list of dicts, use the "content" field.
            input_text = (
                messages[0]["content"] if isinstance(messages[0], dict) and "content" in messages[0] 
                else messages[0]
            )
            response = client.embeddings.create(
                model=model,
                input=input_text
            )
            return response.data[0].embedding

        elif model.startswith("o"):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort="high",
                max_completion_tokens=12_000,
                stream=stream
            )
            if stream:
                complete_output = ""
                for chunk in response:
                    token = chunk.choices
                    if len(token) > 0:
                        token = token[0].delta.content
                    else:
                        token = ""
                    token = "" if token is None else token
                    complete_output += token
                return complete_output
            else:
                return response.choices[0].message.content

        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                stream=stream
            )
            if stream:
                complete_output = ""
                for chunk in response:
                    token = chunk.choices
                    if len(token) > 0:
                        token = token[0].delta.content
                    else:
                        token = ""
                    token = "" if token is None else token
                    complete_output += token
                return complete_output
            else:
                return response.choices[0].message.content

    elif provider == "anthropic":
        if messages[0]["role"] == "system":
            system_message = messages.pop(0)["content"]
        else:
            system_message = None

        kwargs = {
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "messages": messages,
            "model": model,
            "stream": stream,
            **extra_kwargs,
        }
        if "thinking" in kwargs:
            kwargs.pop("top_p")
            kwargs.pop("temperature")

        if system_message is not None:
            kwargs.update({"system": system_message})
                        
        return _anthropic_completion(**kwargs)


def clone_repository(repo_url, clone_dir):
    # Using subprocess to call git clone
    print(f"Cloning repository {repo_url} into {clone_dir}")
    result = subprocess.run(["git", "clone", repo_url, clone_dir], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Git clone failed for repository {repo_url}.\nError: {result.stderr}")
    return clone_dir

def get_query_embedding(query_text):
    try:
        query_text = truncate_text(query_text, "text-embedding-3-small", 7_800)
        response = call_models(
            messages=[{"role": "user", "content": query_text}],
            provider="openai",
            model="text-embedding-3-small",
            max_tokens=8_000,
            temperature=0.2,
        )
        return response
    except Exception as e:
        print(f"Error getting query embedding: {e}")
        return [0] * EMB_DIM

def get_file_embeddings(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error getting file embedding: {e}")
        content = ""

    vector = get_query_embedding(content)
    emb_cols = [f"emb_{i}" for i in range(EMB_DIM)]

    return {
        "file_path": file_path,
        "content": content,
        **dict(zip(emb_cols, vector))
    }

def build_index(repo_dir, index_dir):
    data = []
    file_paths = subprocess.check_output(["git", "ls-files"], cwd=repo_dir).decode("utf-8").strip("\n").strip().split("\n")
    file_paths = [os.path.join(repo_dir, file_path) for file_path in file_paths]

    with tqdm.tqdm(total=len(file_paths), desc="Processing files...") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(get_file_embeddings, file_path) for file_path in file_paths]

            for future in as_completed(futures):
                result = future.result()
                data.append(result)
                pbar.update(1)

    df = pd.DataFrame(data)
    
    # Ensure the cache directory exists
    cache_path = os.path.expanduser(index_dir)
    os.makedirs(cache_path, exist_ok=True)
    csv_path = os.path.join(cache_path, "repo.csv")
    df.to_csv(csv_path, index=False)
    
    return df

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    """
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def retrieve_top_docs(query_vector, df, emb_cols, top_n=10):
    emb_values = df[emb_cols].values
    embeddings = np.array(emb_values)
    query_vector = np.array(query_vector)

    query_norm = np.linalg.norm(query_vector)
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    embedding_norms = np.where(embedding_norms == 0, 1e-10, embedding_norms)

    dot_products = np.dot(embeddings, query_vector)
    similarities = dot_products / (embedding_norms * query_norm)

    df = copy.deepcopy(df)
    df["similarity"] = similarities

    top_docs = df.sort_values(by="similarity", ascending=False).head(top_n)
    return top_docs

def query_llm(model, documents, query):
    combined_content = "\n".join([f"File: {doc['file_path']}\nContent: {doc['content']}" for doc in documents])
    messages = [
        {"role": "system", "content": "You are a Coding Assistant."},
        {"role": "user", "content": f"Here is the combined content of the documents: {combined_content}\n\nBased on that answer the below query.\n\nQuery: {query}"}
    ]

    if model.startswith("gpt"):
        response = call_models(
            messages=messages,
            provider="openai",
            model="gpt-4o",
            max_tokens=16384,
            temperature=0.2,
            stream=True,
        )
    elif model.startswith("claude"):
        response = call_models(
            messages=messages,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            max_tokens=16384,
            temperature=0.2,
            stream=True,
        )

    return response


# Centered title using markdown and HTML
st.markdown("<h1 style='text-align: center;'>Coding Assistant</h1>", unsafe_allow_html=True)

# Dummy options for models and repositories
models = ["gpt-4o", "claude-3-5-sonnet-20241022"]
repositories = [
    "https://github.com/sourcegraph/sourcegraph-public-snapshot",
    "https://github.com/huggingface/transformers",
    "https://github.com/pytorch/pytorch",
    "https://github.com/keras-team/keras-nlp",
]

# UI elements
selected_model = st.selectbox("Select Model:", models)
selected_repo = st.selectbox("Select Repository:", repositories)
query = st.text_area("Enter your query:", height=150)

# Center the button using CSS
st.markdown(
    """
    <style>
    div.stButton > button {
        display: block;
        margin: 0 auto;
        padding: 0.5rem 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("Send"):
    if not selected_repo or not query:
        st.warning("Please enter a query and select a repository.")
    else:
        st.info(f"Starting to process your query over the selected repository...")
        
        repo_dir = os.path.join(TARGET_DIR, "repos/", selected_repo.removeprefix("https://github.com/").removesuffix(".git").replace("/", "_"))
        index_dir = os.path.join(TARGET_DIR, "indices/", selected_repo.removeprefix("https://github.com/").removesuffix(".git").replace("/", "_"))

        if not os.path.exists(repo_dir):
            try:
                repo_dir = clone_repository(selected_repo, repo_dir)
                st.write("Repository cloned successfully!")
            except Exception as e:
                st.error("Error cloning repository: " + str(e))
                st.stop()
        else:
            if not os.path.exists(index_dir):
                try:
                    df_index = build_index(repo_dir, index_dir)
                    st.write(f"Index built and stored at {index_dir}")
                except Exception as e:
                    st.error("Error building index: " + str(e))
                    st.stop()
        
        if os.path.exists(index_dir):
            # load it
            df_index = pd.read_csv(os.path.join(index_dir, "repo.csv"))
            query_vector = get_query_embedding(query)
            emb_cols = [f"emb_{i}" for i in range(EMB_DIM)]
            top_docs = retrieve_top_docs(query_vector, df_index, emb_cols, top_n=10)
            
            st.subheader("Top Retrieved Documents:")
            st.write(top_docs[["file_path", "similarity"]])
            
            # Gather contents from the top documents
            doc_contents = top_docs[["file_path", "content"]].to_dict(orient="records")
            
            # Query the LLM using the combined document content and query
            answer = query_llm(selected_model, doc_contents, query)
            
            st.subheader("Final Response:")
            st.write(answer)

