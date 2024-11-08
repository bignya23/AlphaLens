import requests
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# Loading the environment variable
load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

#Function to send_request
def send_request(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        raise EnvironmentError("GEMINI_API_KEY is not set in the environment.")
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=8192,
        timeout=None,
        max_retries=2,
        top_k=40,
        top_p=0.95
        )

    chain = prompt | llm

    output = chain.invoke(
        {
            "input": prompt,
        }
    )

    return output.content


def join_headlines(headlines_df):

    # Concatenate the headlines
    headlines_text = ". ".join(headlines_df['title'].tolist())

    return headlines_text

def bullish(headlines_df):

    headlines_text = join_headlines(headlines_df)


    prompt_bullish = ChatPromptTemplate.from_template(
    f"""
    Analyze these news headlines and provide a 'Bullish Insights' summary, focusing on positive trends or growth areas.
    Limit to around 100 words.
    
    Headlines:
    {headlines_text}

    Keep it concise and actionable.
    ouptut
    {input}
    """
)




    bullish_summary = send_request(prompt_bullish)

    return bullish_summary

def bearish(headlines_df):

    headlines_text = join_headlines(headlines_df)

    prompt_bearish = ChatPromptTemplate.from_template(
        f"""
        Analyze the following headlines to provide an 'Investment Insights' summary.
        Highlight short-term and long-term indicators on whether it might be a good time to invest.
        Mention key risks and growth areas in 100 words or less.

        Headlines:
        {headlines_text}
           ouptut
           {input}
           """


    )

    bearish_summary = send_request(prompt_bearish)

    return bearish_summary

#Function to generate investment insights
def investment_insights(headlines_df):

    headlines_text = join_headlines(headlines_df)


    prompt_summary = ChatPromptTemplate.from_template(
    f"""
    Analyze the following headlines to provide an 'Investment Insights' summary.
    Highlight short-term and long-term indicators on whether it might be a good time to invest.
    Mention key risks and growth areas in 300 words or less.

    Headlines:
    {headlines_text}

    Output the insights in paragraph format.
    output:
    {input}
    """
)



    investment_insight = send_request(prompt_summary)

    return investment_insight
