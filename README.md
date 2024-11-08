#  WEBSITE LINK : https://alphalens.tech




# AlphaLens: A Functional Platform for Financial News Classification and Sentiment Analysis

**AlphaLens** is a comprehensive platform that classifies financial news articles, performs sentiment analysis, and provides real-time investment insights. The platform integrates machine learning models with a user-friendly interface, offering essential financial data, sentiment classifications, and visual market trends for investors and financial analysts.

## Key Features

- **User-Friendly Interface**: Enter a company name, and AlphaLens fetches the corresponding stock symbol and provides:
  - Investment insights, including major and institutional holders
  - Analysts' target prices and recent news headlines with sentiment classifications
  - An embedded TradingView chart for stock performance visualization

- **Classification Engine**: Utilizes FinBert, a financial-specific NLP model, to classify news sentiment (positive, negative, or neutral) for targeted and accurate analysis.

- **Sentiment Analysis Module**: Powered by the Gemini 1.5 - Flash model,  this module summarizes news articles and provides in-depth market insights.

- **Real-Time Processing**: News articles and financial data are processed in real-time, keeping insights current and reliable.

- **Data Visualization**: Sentiment trends, classification results, and TradingView charts are displayed through dynamic dashboards for an enhanced user experience.

## Technical Overview

### System Architecture

The platform architecture is designed for efficiency and scalability, integrating a real-time backend and an intuitive frontend.

- **Frontend (Svelte)**: A dynamic, responsive interface that allows users to search by company name and view organized financial insights.

- **Backend (Flask and Socket.IO)**: Handles data processing, API requests, and real-time data updates using Socket.IO to provide users with up-to-date insights.

- **Company Symbol Database**: Maps company names to stock symbols via a pre-integrated database on Vultr, making searches fast and user-friendly.

- **News and Financial Data Retrieval**:
  - **NewsAPI**: Fetches relevant news articles for sentiment analysis.
  - **yFinance**: Provides fundamental stock data, including information on holders and financial ratios.

- **Sentiment Analysis and Summarization**:
  - **FinBert**: Classifies news sentiment as positive, negative, or neutral.
  - **Alphalens** leverages the Gemini 1.5 - Flash model for advanced sentiment analysis and natural language understanding, enhancing the accuracy of financial news classification and sentiment detection.

### Deployment and Infrastructure

AlphaLens is hosted on Vultr Cloud Compute, optimized for scalability, security, and performance:

- **DNS Management and Firewall**: Vultr DNS and firewall configurations ensure secure, reliable data access and content delivery.
- **Pull and Push Zones**:
  - **Pull Zone**: Caches content for faster access.
  - **Push Zone**: Ensures fresh data distribution for real-time updates.

### Key Modules

- **yFinance**: Retrieves advanced stock data, providing insights into financial performance and ratios.
- **FinBert**: Customizes sentiment classification for financial news, providing relevant sentiment data to users.

## API Documentation

### NewsAPI Integration

Alphalens uses [NewsAPI](https://newsapi.org/) to pull real-time financial news from a range of reputable sources. The "Top Headlines" endpoint allows for targeted searches by company name, keyword, and other filters, providing timely news content for sentiment analysis. For detailed usage and authentication guidelines, refer to the [NewsAPI documentation](https://newsapi.org/docs).

---




# AlphaLens

AlphaLens is a sentiment analysis platform that provides insights into company news, stock sentiment, and fundamental analysis.

## Setup and Usage Instructions

### Prerequisites
Ensure you have the following installed:
- **Python** (3.7 or higher)
- **pip** (Python package installer)
- **Node.js** and **npm** (Node Package Manager)

### Step 1: Clone the Repository
Clone the Alphalens GitHub repository to your local machine. Open a terminal or command prompt and run:
```bash
https://github.com/bignya23/AlphaLens.git
```


### Step 2: Install Python Requirements
Navigate to the project folder and install the required Python dependencies:

1. Move to the Alphalens folder:
    ```bash
    cd Alphalens
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Step 3: Set Up Environment Variables
Create a `.env` file in the `backend` folder with all the required API keys and other configurations. Refer to the repository’s documentation or README for the specific variables needed.
## Environment Variables

To set up the environment for this project, add the following variables to a `.env` file in the `backend` folder. Each variable is described below:

```env
# API key for accessing the News API to fetch financial news and data for the application.
NEWS_API_KEY=your_news_api_key_here

# Hugging Face API token for accessing NLP models or datasets from Hugging Face.
hf_token=your_hugging_face_token_here

# Gemini api token for assessing gemini 1.5 flash
GEMINI_API_KEY = your_gemini_api_here

#Langchain api token to integrate langchain_services
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY= your_langchain_api_key
LANGCHAIN_PROJECT = your_project_name

# API key for interacting with Vultr cloud services.
VULTR_API=your_vultr_api_key_here

# PostgreSQL database connection URL, which provides access to the application’s primary database.
DATABASE_URL=your_database_url_here
```

### Step 4: Run the Backend
1. Move to the backend folder:
    ```bash
    cd backend
    ```

2. Start the backend server:
   
  For Windows:
  
  ```bash
    python app.py
  ```
  

For Mac/Unix:
  ```bash
    export FLASK_APP=app.py
    export FLASK_ENV=development
    flask run

  ```

### Step 5: Set Up and Run the Frontend
1. Navigate back to the main directory:
    ```bash
    cd ..
    ```

2. Move to the frontend folder:
    ```bash
    cd frontend
    ```

3. Install the necessary Node.js dependencies:
    ```bash
    npm install
    ```

4. Start the frontend development server:
    ```bash
    npm run dev
    ```

The website should now be accessible on your local server. Check the terminal output for the exact URL (typically `http://localhost:5173`).

## Usage Instructions
- **Select a Company**: Once the website is running, go to the top search bar and click on "Select Company." This will display a list of all S&P 500 companies. Choose a company to analyze.
  
- **Chatbot Assistance**: Use the chatbot located in the bottom right corner to ask any finance or stock market-related questions. The chatbot can provide insights and help with your analysis.

