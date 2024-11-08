import requests
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

def generate_summary_and_insights_from_fundamentals(stock_info):
    # Extracting major holders, institutional, mutual fund, insider transactions, and earnings/revenue estimates
    major_holders = stock_info.get('Major Holders', 'No data available')
    institutional_holders = stock_info.get('Institutional Holders', 'No data available')
    mutual_fund_holders = stock_info.get('Mutual Fund Holders', 'No data available')
    insider_transactions = stock_info.get('Insider Transactions', 'No data available')
    insider_purchases = stock_info.get('Insider Purchases', 'No data available')
    analyst_price_targets = stock_info.get('Analyst Price Targets', 'No data available')
    earnings_estimate = stock_info.get('Earnings Estimate', 'No data available')
    revenue_estimate = stock_info.get('Revenue Estimate', 'No data available')

    # Formulating a concise prompt
    prompt = ChatPromptTemplate.from_template(
            "Analyze the following stock's fundamental data and provide a summary:\n\n"

            f"### Major Holders\n{major_holders}\n\n"

            f"### Institutional Holders (Top 10)\n"
            + "\n".join([
        f"- {row['Holder']}: {row['pctHeld']} of shares held."
        for index, row in institutional_holders.head(10).iterrows()
    ]) + "\n\n"

         f"### Mutual Fund Holders (Top 10)\n"
            + "\n".join([
        f"- {row['Holder']}: {row['pctHeld']} of shares held."
        for index, row in mutual_fund_holders.head(10).iterrows()
    ]) + "\n\n"

         f"### Insider Transactions\n{insider_transactions.head(3).to_string(index=False)}\n\n"

         f"### Insider Purchases\n"
         f"- Purchases: {insider_purchases.get('Purchases', 'N/A')}\n"
         f"- Sales: {insider_purchases.get('Sales', 'N/A')}\n\n"

         f"### Analyst Price Targets\n"
         f"Current Price Target: ${analyst_price_targets['current']}\n"
         f"Mean Price Target: ${analyst_price_targets['mean']}\n\n"

         f"### Earnings Estimate (Next Quarter)\n"
            + f"- Average EPS: {earnings_estimate['avg'].iloc[0]}\n"
            + f"- Growth: {earnings_estimate['growth'].iloc[0]}\n\n"

              f"### Revenue Estimate (Next Quarter)\n"
            + f"- Average Revenue: {revenue_estimate['avg'].iloc[0]}\n"
            + f"- Growth: {revenue_estimate['growth'].iloc[0]}\n"
            
            "Output"
            f"{input}"
    )


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