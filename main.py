import os 
import asyncio
import shutil
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import agentic_setup

def storing_markdown_output(native_md_response: str):
    """
    Stores the markdown output to a file named 'native_agentic_output.md'.
    If the file already exists, it appends the new content to the end of the file.
    """
    file_name = "native_agentic_output.md"

    dir_path = "markdown_outputs"

    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, file_name)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(native_md_response + "\n")
    print(f"Markdown output saved to {file_path}")

def initialize_model(model_name: str, model_type: str) -> object:
    """
    Initializes the model based on the provided model name.
    Currently supports 'gemini' for Google Gemini and 'huggingface' for Hugging Face models.
    """
    if model_name == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        raw_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=1,
            api_key=gemini_key
        )
        
    
    elif model_name == "huggingface":
        hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_key:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")
        repoid = "Qwen/Qwen3-4B-MLX-bf16:featherless-ai"
        raw_llm = HuggingFaceEndpoint(
            repo_id=repoid,
            task="text-generation",
            max_new_tokens=1500,
            temperature=0.
        )

    elif model_name == "groq":
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is not set in the environment variables.")
        raw_llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=None,
            temperature=0
        )
        
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return raw_llm

async def main(model_search, model_parsing):
    # 0. MCP server setup and tool loading 

    ddgs_path = shutil.which("ddgs")
    
    if ddgs_path is None:
        raise FileNotFoundError("Python cannot find 'ddgs' anywhere in your system paths. Please ensure 'pip install -U ddgs[mcp]' was run in this exact environment.")
    
    print(f"Found 'ddgs' at: {ddgs_path}")

    client = MultiServerMCPClient({
        "ddgs": {
            "transport": "stdio",
            "command": ddgs_path,
            "args": ["mcp"]
        }
    }) 
    async with client.session("ddgs") as session:
        mcp_tools = await load_mcp_tools(session)

        agent_search = model_search.bind_tools(mcp_tools, tool_choice="required")
    
        raw_content = await agentic_setup.reserch_agent(agent_search, "Adwisely", "https://adwisely.com")
        structured_content = await agentic_setup.parsing_agent(model_parsing, raw_content)
        
        parsed = agentic_setup.assemble_markdown_output(structured_content)
        print(f"Final Parsed Content: {parsed}")



if __name__ == "__main__":

    load_dotenv() 
    model_search = initialize_model("groq", "search")
    model_parsing = initialize_model("groq", "parsing")
    asyncio.run(main(model_search, model_parsing))

   


