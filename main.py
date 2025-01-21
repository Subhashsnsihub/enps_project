import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SERVICE_ACCOUNT_FILE = r'C:\Users\HP\Desktop\eNPS\gen-lang-client-0660452616-7f8dc01be13a.json'
SHEET_ID = '10ET8bZJ5GbTS6oCb8Qdq4uUoSV7bgNwPEYhoELOQhNA'  
try:
    
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    
    
    result = service.spreadsheets().values().get(
        spreadsheetId=SHEET_ID, 
        range="Sheet1!A1:E81"
    ).execute()
    
    # Print data
    values = result.get('values', [])
    if not values:
        print("No data found.")
    else:
        for row in values:
            print(row)
        
except FileNotFoundError:
    print("Service account file not found.")
except Exception as e:
    print(f"Error: {e}")

import pandas as pd

try:
    if not values:
        print("No data to process.")
    else:
        # Convert to DataFrame
        headers = values[0]  # First row as column headers
        data = values[1:]    # Remaining rows as data
        df = pd.DataFrame(data, columns=headers)

        print("Data loaded into DataFrame:")
        print(df.head())  
except Exception as e:
    print(f"Error during DataFrame creation: {e}")


# Fill missing values and convert data types
df['Tenure (Months)'] = df['Tenure (Months)'].fillna(0).astype(int) 
df['Feedback'] = df['Feedback'].fillna('No Feedback')  
df['Department'] = df['Department'].str.strip().str.lower()  

# Create derived features
df['Tenure_Years'] = df['Tenure (Months)'] / 12  

# One-hot encode the Department column
df_encoded = pd.get_dummies(df, columns=['Department'], prefix='Dept', drop_first=True)

# Clean feedback text
df_encoded['Feedback_Cleaned'] = df_encoded['Feedback'].str.lower().str.strip()

# Save cleaned data
output_path = "C:/Users/HP/Desktop/eNPS/cleaned_eNPS_data.xlsx"
df_encoded.to_excel(output_path, index=False)
print(f"Transformed dataset saved to {output_path}")

# Convert boolean values in department columns to numeric (1/0)
dept_columns = [col for col in df_encoded.columns if col.startswith('Dept_')]
for col in dept_columns:
    df_encoded[col] = df_encoded[col].map({True: 1, False: 0})

# Clean eNPS Score - convert to numeric
df_encoded['eNPS Score'] = pd.to_numeric(df_encoded['eNPS Score'], errors='coerce')

# Verify the department encoding
print("Sample of encoded data:")
print(df_encoded[dept_columns].head())


from transformers import pipeline


sentiment_analyzer = pipeline("sentiment-analysis")


def get_sentiment(text):
    if isinstance(text, str) and text != 'no feedback':
        
        sentiment = sentiment_analyzer(text)
        # Return the sentiment score (0 for neutral, positive/negative scores)
        return sentiment[0]['score'] if sentiment[0]['label'] == 'POSITIVE' else -sentiment[0]['score']
    return 0
df_encoded['Sentiment_Score'] = df_encoded['Feedback_Cleaned'].apply(get_sentiment)

# Display the updated dataframe with sentiment scores
print(df_encoded[['Feedback_Cleaned', 'Sentiment_Score']].head())


# Categorize eNPS scores with Hugging Face or sentiment-based analysis if needed
def categorize_enps(score, sentiment_score=None):
 
    if pd.isna(score) or score == 0:
        return 'No Score'
    
    # You can include sentiment_score influence if you'd like to add a layer of sentiment-based categorization
    elif score >= 9:
        return 'Promoter'
    elif score >= 7:
        return 'Passive'
    else:
        return 'Detractor'

df_encoded['eNPS_Category'] = df_encoded.apply(lambda row: categorize_enps(row['eNPS Score'], row['Sentiment_Score']), axis=1)

# Check the updated categories
print(df_encoded[['eNPS Score', 'Sentiment_Score', 'eNPS_Category']].head())


# Calculate overall eNPS using the categorized data
promoters = (df_encoded['eNPS_Category'] == 'Promoter').mean()
detractors = (df_encoded['eNPS_Category'] == 'Detractor').mean()

print(promoters,'promoters_avg')
print(detractors,'detractors_avg')
# Calculate eNPS score
enps = (promoters - detractors) * 100

# Basic analysis (overall eNPS)
print(f"Overall eNPS Score: {enps:.1f}")
print("\nDistribution of eNPS Categories:")
print(df_encoded['eNPS_Category'].value_counts())

# Department-wise analysis
print("\nAverage eNPS Score by Department:")
for col in dept_columns:
    dept_name = col.replace('Dept_', '')
    # Average eNPS score for each department based on the 'eNPS Score'
    avg_score = df_encoded[df_encoded[col] == 1]['eNPS Score'].mean()
    print(f"{dept_name}: {avg_score:.1f}")




# Save enhanced analysis with sentiment and categorized eNPS data
output_path = "C:/Users/HP/Desktop/eNPS/analyzed_eNPS_data.xlsx"
df_encoded.to_excel(output_path, index=False)
print(f"\nAnalyzed dataset saved to {output_path}")


from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
texts = df_encoded['Feedback_Cleaned'].values  # or another column containing feedback

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
X = vectorizer.fit_transform(texts)

# Fit LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)  # Choose number of topics
lda.fit(X)

# Create word clouds for each topic
for index, topic in enumerate(lda.components_):
    print(f"Topic #{index + 1}:")
    words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]  # top 10 words for the topic
    word_freq = {word: topic[i] for i, word in enumerate(words)}  # Word frequencies for the cloud

    # Generate the WordCloud
    wordcloud = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(word_freq)

    # Display the word cloud
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Topic #{index + 1}")
    plt.show()


