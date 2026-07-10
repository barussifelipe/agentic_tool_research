import os 
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
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





if __name__ == "__main__":

    load_dotenv() 

    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")    

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=1
        )

    native_md_response = agentic_setup.run_native_agent_pipeline(model, "Getscreen.me", "http://getscreen.me")

    storing_markdown_output(native_md_response)
