import os
import json
import sys
from typing import List
from pydantic import BaseModel, Field 
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
# ==========================================
# DEFINE THE STRICT DATA SCHEMA
# ==========================================
class AppDataSchema(BaseModel):
    name: str = Field(description="Official product name")
    tagline: str = Field(description="Max 35 characters, short description")
    slug: str = Field(description="Lowercase, hyphenated (e.g. hubspot-crm)")
    category: str = Field(description="Semicolon separated categories matching permitted options")
    description: str = Field(description="~200 word description of the flagship product")
    best_for: str = Field(description="~200 word description of who the tool is best suited for. Use this structure: 5 bulletpoints and then empty row and then simple text for 'Not ideal for:'")
    setup_complexity: str = Field(description="~200 word description. Use this structure: first row simple text with general duration of implementation. empty row then 5 bulletpoints. empty row and then Complexity rating")
    website: str = Field(description="Official website URL")
    affiliate_url: str = Field(default="")
    pricing_model: str = Field(description="Freemium; Free; Pay-as-you-go; Per-seat; Per-contractor; Transaction-based; Flat-rate; Custom")
    starting_price: str = Field(description="'Free version'; 'Custom pricing'; or a specific price like '€9/mo'")
    amount: float = Field(default=0.0, description="Number only matching official pricing currency")
    amount_yearly: float = Field(default=0.0, description="Number only for yearly billing monthly equivalent")
    currency: str = Field(default="EUR", description="Must be in the original currency of the official pricing page (e.g., EUR, USD, GBP)")
    pros: str = Field(description="5-7 key strengths separated by '; '")
    cons: str = Field(description="3-5 weaknesses separated by '; '")
    integrations: str = Field(description="Direct native app integrations separated by '; ' ordered by website order of appearance.")
    alternatives: str = Field(description="Competing products separated by '; '")
    platforms: str = Field(description="Web; Mac; Windows; iOS; Android; Linux separated by '; '")
    flags: str = Field(default="")
    deployment: str = Field(description="Cloud; Hybrid; On-Premise separated by '; '")
    role_categories: str = Field(description="Entrepreneurs; Managers; Freelancers; Developers; Operations; Students; Designers")
    company_size: str = Field(description="1-10; 11-50; 51-200; 200+ separated by '; '")
    su_category: str = Field(description="Core; HR; Data & Finance; Sales & Marketing; Product; Ops, Tools & Automations; Velocity; AI / New Tech; Network Leverage")
    logo_url: str = Field(default="", description="Direct URL to PNG/SVG logo")
    free_version_and_trial: str = Field(description="free_version; free_trial; both; none")
    pricing_link: str = Field(description="Direct URL to pricing page")
    resell_info_page: str = Field(default="")
    resell_apply_form_call: str = Field(default="")
    reseller_management_portal: str = Field(default="")
    payout_gate: str = Field(default="")
    pipeline_kanban: str = Field(default="")
    country: str = Field(description="Origin country")
    city: str = Field(description="Origin city")
    scaling_notes: str = Field(description="Max 800 chars. Scalable tiered pricing structure outlining plan costs (monthly and anually billing), matching the currency in the OFFICIAL pricing page, and included features.")
    linkedin_link: str = Field(default="")
    youtube_channel_link: str = Field(default="")
    instagram_page_link: str = Field(default="")
    twitter_x_page_link: str = Field(default="")
    
    # Raw source links separated for the final layout agent
    source_prices_scaling: List[str] = Field(description="List of URLs used for pricing")
    source_tiers: List[str] = Field(description="List of tier strings like 'Tier1: PRICE'")
    source_socials: List[str] = Field(description="List of social media source links")
    source_integrations_page: str = Field(description="URL to native integrations page")


# ==========================================
# AGENT DEFINITIONS
# ==========================================


async def reserch_agent(agent, app_name: str, app_website: str) -> str:

    research_prompt = f"""
    Perform deep research on the company and tool: {app_name}. Core Website: {app_website}.
    Locate their official pricing page, integration directory, corporate headquarters location, and social profiles.
    CRITICAL FLAGSHIP RULE: If this brand has multiple products, identify the SINGLE highest-revenue/flagship product.
    Extract data exclusively for this flagship product. 
    Dump all raw text details, specific pricing numbers, tiers, features, and direct URLs.
    """
    
    result = await agent.ainvoke([
            
                ("system", "You are a research expert."
                    "CRITICAL INSTRUCTION: You MUST use your search tools to gather up-to-date information FIRST. "
                    "DO NOT answer from your internal memory or training data. "
                    "DO NOT write any summaries or reports until you have successfully executed a web search and read the results."),
                ("user", research_prompt)]
    )
    print(f"Tokens used for research agent: {result.token_usage.usage_metadata.total_tokens}, input tokens: {result.token_usage.usage_metadata.input_tokens}, output tokens: {result.token_usage.usage_metadata.output_tokens}")
    print(f"Tools called during research agent execution: {result.tool_calls}")
    return result["messages"][-1].content 

async def parsing_agent(agent, raw_context: str) -> str:
        
    parser = PydanticOutputParser(pydantic_object=AppDataSchema)
    formatted_instructions = parser.get_format_instructions()
    print(f"Formatted Instructions for Parsing Agent: {formatted_instructions}")

    structurer_prompt = """
    "You are a Structurer Agent with expertise in SaaS product analysis. Your task is to extract and structure data into a strict JSON schema. \n 
    Take the raw data payload below and map it strictly into the required JSON schema structure.\n
    
    CRITICAL RULES:
    1. Focus only on the main flagship product found in the context.\n
    2. Currency MUST be mapped as the one found on the official pricing page. Do not convert numerical figures; extract numerical amounts exactly as seen on the official page.\n
    3. Respect text format requirements (e.g., specific structures for 'Best For' and 'Setup Complexity').\n
    4. Integrations must strictly be direct, native integrations found in the text. Do not list Zapier libraries.\n
    --- Format ---
    {formatted_instructions}\n\n
    --- Raw Data Context to Parse ---:\n
    {text}\n
    """

    prompt = PromptTemplate(
        template = structurer_prompt,
        input_variables=["text"], 
        partial_variables={"formatted_instructions": formatted_instructions},
    )

    extraction_chain = prompt | agent | parser 

    structured_output = await extraction_chain.ainvoke({"text": raw_context})
    print(structured_output)

    return structured_output


def assemble_markdown_output(data: dict) -> str:
    headers = [
        "Name", "Tagline", "Slug", "Category", "Description", "Best For", "Setup Complexity", "Website", 
        "Affiliate URL", "Pricing Model", "Starting Price", "Amount", "Amount Yearly", "Currency", "Pros", 
        "Cons", "Integrations", "Alternatives", "Platforms", "Flags", "Deployment", "Role Categories", 
        "Company Size", "SU.category", "Logo URL", "Free Version & Trial", "Pricing Link", "Resell Info Page", 
        "Resell Apply Form/Call", "Reseller Management Portal", "Payout Gate", "Pipeline Kanban", "Country", "City"
    ]
    
    # Build Row 1 (Core 34 Columns + Extended Properties Appended appropriately as needed or mapped cleanly)
    # Mapping exact spreadsheet keys
    row_values = [
        data.get("name", ""), data.get("tagline", ""), data.get("slug", ""), data.get("category", ""),
        data.get("description", ""), data.get("best_for", ""), data.get("setup_complexity", ""), data.get("website", ""),
        data.get("affiliate_url", ""), data.get("pricing_model", ""), data.get("starting_price", ""), f"{data.get("amount", 0.0):.2f}",
        f"{data.get("amount_yearly", 0.0):.2f}", data.get("currency", "EUR"), data.get("pros", ""), data.get("cons", ""),
        data.get("integrations", ""), data.get("alternatives", ""), data.get("platforms", ""), data.get("flags", ""),
        data.get("deployment", ""), data.get("role_categories", ""), data.get("company_size", ""), data.get("su_category", ""),
        data.get("logo_url", ""), data.get("free_version_and_trial", ""), data.get("pricing_link", ""), data.get("resell_info_page", ""),
        data.get("resell_apply_form_call", ""), data.get("reseller_management_portal", ""), data.get("payout_gate", ""),
        data.get("pipeline_kanban", ""), data.get("country", ""), data.get("city", "")
    ]
    
    # Clean cell values of any tabs or line breaks to preserve formatting integrity
    row_values = [str(val).replace("\t", " ").replace("\n", " ") for val in row_values]
    
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    markdown_table += "| " + "\t".join(row_values) + " |\n" # User requested columns separated by tabs within row
    
    # Source Section Assembly
    sources_output = f"""
{markdown_table}

##SOURCE LIST FORMAT:
Prices & Scaling Notes:
"""
    for src in data.get("source_prices_scaling", []):
        sources_output += f"- {src}\n"
        
    sources_output += "###TIERS AND PRICES:\n"
    for tier in data.get("source_tiers", []):
        sources_output += f"- {tier}\n"
        
    sources_output += "###Social media links:\n"
    sources_output += f"- LinkedIn: {data.get('linkedin_link', '')}\n"
    sources_output += f"- YouTube: {data.get('youtube_channel_link', '')}\n"
    sources_output += f"- Instagram: {data.get('instagram_page_link', '')}\n"
    sources_output += f"- Twitter (X): {data.get('twitter_x_page_link', '')}\n\n"
    
    sources_output += "###Integrations:\n"
    sources_output += f"- Integrations page: {data.get('source_integrations_page', '')}\n"
    
    return sources_output