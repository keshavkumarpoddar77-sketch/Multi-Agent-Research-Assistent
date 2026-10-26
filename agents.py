from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import config
from tools import web_search, scrape_url

load_dotenv()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model=config.MODEL_NAME,
    temperature=config.TEMPERATURE,
    timeout=config.TIMEOUT,
    max_retries=config.MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def build_search_agent():
    return create_agent(model=llm, tools=[web_search])


def build_reader_agent():
    return create_agent(model=llm, tools=[scrape_url])


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
WRITER_RULES = """Rules:
- Cite sources inline like [1], [2] after every factual claim.
- Include specific facts, numbers and examples from the research.
- Never invent statistics, names or URLs. If the research does not say it, leave it out.
- Each key finding needs at least two sentences of explanation.
- End with a numbered list of sources that matches your inline citations."""

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below. Length: about {words} words.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources

""" + WRITER_RULES),
]).partial(words=config.REPORT_WORDS)

writer_chain = writer_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest, strict and specific."),
    ("human", """Review the research report below and evaluate it strictly.
Judge accuracy, use of sources, depth, structure and clarity.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Reviser (rewrites the report using the critic's feedback)
# ---------------------------------------------------------------------------
revise_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer revising a report using critic feedback."),
    ("human", """Topic: {topic}

Research:
{research}

Current report:
{report}

Critic feedback:
{feedback}

Rewrite the full report and fix every point under "Areas to Improve".
Keep the same structure (Introduction, Key Findings, Conclusion, Sources). Length: about {words} words.

""" + WRITER_RULES),
]).partial(words=config.REPORT_WORDS)

revise_chain = revise_prompt | llm | StrOutputParser()