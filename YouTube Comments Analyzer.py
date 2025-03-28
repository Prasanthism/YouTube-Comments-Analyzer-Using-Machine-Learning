import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from wordcloud import WordCloud
import seaborn as sns
import torch

class YouTubeSentimentAnalysis:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_all_video_comments(self, video_id):
        comments = []
        youtube = build('youtube', 'v3', developerKey=self.api_key)
        nextPageToken = None

        while True:
            try:
                comments_response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    textFormat='plainText',
                    pageToken=nextPageToken
                ).execute()

                for comment_item in comments_response.get('items', []):
                    comment = comment_item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comment = self.preprocess_comment(comment)
                    comments.append(comment)

                nextPageToken = comments_response.get('nextPageToken')
                if not nextPageToken:
                    break

                time.sleep(1)  

            except HttpError as e:
                print(f"HTTP Error: {e}")
                if e.resp.status in [403, 400]:
                    print("Rate limit exceeded or invalid request. Try again later.")
                    break
                time.sleep(5)  

            except ConnectionResetError:
                print("Connection lost. Retrying...")
                time.sleep(5)
        
        return comments

    def preprocess_comment(self, comment):
        comment = re.sub(r'[^\w\s]', '', comment)
        comment = comment.encode('ascii', 'ignore').decode('ascii')
        return comment

    def analyze_sentiments(self, video_id):
        comments = self.get_all_video_comments(video_id)
        if not comments:
            print("No comments fetched. Exiting...")
            return [], []

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        sentiment_scores = []
        for comment in comments:
            inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()
            sentiment_scores.append(self.classify_sentiment(probabilities))
        
        return comments, sentiment_scores

    def classify_sentiment(self, probabilities):
        neutral_prob, positive_prob = probabilities 
        if neutral_prob > 0.4 and neutral_prob < 0.6:
            return "Neutral"
        elif positive_prob > 0.5:
            return "Positive"
        else:
            return "Negative"

class SentimentAnalysisVisualizer:
    def create_pie_chart(self, sentiment_scores):
        unique_sentiments, sentiment_counts = np.unique(sentiment_scores, return_counts=True)

        
        sentiment_dict = {"Positive": 0, "Negative": 0, "Neutral": 0}

        
        for sentiment, count in zip(unique_sentiments, sentiment_counts):
            sentiment_dict[sentiment] = count

        labels = list(sentiment_dict.keys())
        counts = list(sentiment_dict.values()) 
        colors = ['green', 'red', 'blue']

        plt.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
        plt.title('Sentiment Distribution')
        plt.show()

    def create_bar_chart(self, sentiment_scores):
        unique_sentiments, counts = np.unique(sentiment_scores, return_counts=True)

        sns.barplot(x=unique_sentiments, y=counts, hue=unique_sentiments, palette={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}, legend=False)
        plt.xlabel("Sentiment")  
        plt.ylabel("Count")
        plt.title("Sentiment Analysis of Comments")
        plt.show()

    def create_word_cloud(self, comments):
        text = " ".join(comments)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Comments")
        plt.show()

if __name__ == '__main__':
    api_key = "AIzaSyB3au_OC62KowuP2DSFT0fAmKa3BTE7Ipw"
    video_id = "uDg78JwQltY"                   

    youtube_analysis = YouTubeSentimentAnalysis(api_key)
    comments, sentiment_scores = youtube_analysis.analyze_sentiments(video_id)

    if comments and sentiment_scores:
        sentiment_visualizer = SentimentAnalysisVisualizer()
        sentiment_visualizer.create_pie_chart(sentiment_scores)
        sentiment_visualizer.create_bar_chart(sentiment_scores)
        sentiment_visualizer.create_word_cloud(comments)
