"""LangChain chain that converts raw markdown into structured context."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Settings
from .schemas import ContextSummaryModel


CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You convert uploaded database briefing markdown into a compact "
                "structured schema object.\n\n"
                "Rules:\n"
                "- Extract only facts supported by the markdown\n"
                "- Do not invent tables or columns\n"
                "- Keep descriptions concise\n"
                "- Preserve relationship rules and SQL constraints\n"
                "- Dialect must be explicit"
            ),
        ),
        (
            "human",
            (
                "Analyze this database context markdown and return the structured output.\n\n"
                "<context_markdown>\n"
                "{context_markdown}\n"
                "</context_markdown>"
            ),
        ),
    ]
)


def build_context_chain(settings: Settings):
    """Build a runnable chain that returns a ContextSummaryModel."""
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model_id,
        google_api_key=settings.google_api_key,
        temperature=settings.gemini_temperature,
    )
    return CONTEXT_PROMPT | llm.with_structured_output(ContextSummaryModel)
