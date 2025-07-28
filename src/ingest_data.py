import praw
import pandas as pd
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file for local development
load_dotenv()

# --- Configuration ---
# These values control how many posts are fetched. They can be adjusted as needed.
POSTS_PER_SFW_SUB = 100
POSTS_PER_NSFW_SUB = 150 
POSTS_TO_SCAN_PER_MIXED_SUB = 200 

# --- Get Reddit API Credentials from Environment Variables ---
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

if not all([client_id, client_secret, user_agent]):
    raise ValueError("Reddit API credentials (CLIENT_ID, CLIENT_SECRET, USER_AGENT) are not set in the environment.")

# Initialize the Reddit instance
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

# --- Load Subreddit Lists from Environment Variables ---
# This makes the script configurable without changing the code.
SFW_SUBREDDITS_STR = os.getenv("SFW_SUBREDDITS", "")
NSFW_SUBREDDITS_STR = os.getenv("NSFW_SUBREDDITS", "")
MIXED_CONTENT_SUBREDDITS_STR = os.getenv("MIXED_CONTENT_SUBREDDITS", "")

if not all([SFW_SUBREDDITS_STR, NSFW_SUBREDDITS_STR, MIXED_CONTENT_SUBREDDITS_STR]):
    raise ValueError("One or more subreddit list variables (SFW_SUBREDDITS, NSFW_SUBREDDITS, MIXED_CONTENT_SUBREDDITS) are not set in the environment.")

# Convert the comma-separated strings into clean Python lists
SFW_SUBREDDITS = list(set([sub.strip() for sub in SFW_SUBREDDITS_STR.split(',')]))
NSFW_SUBREDDITS = list(set([sub.strip() for sub in NSFW_SUBREDDITS_STR.split(',')]))
MIXED_CONTENT_SUBREDDITS = list(set([sub.strip() for sub in MIXED_CONTENT_SUBREDDITS_STR.split(',')]))


def get_classified_posts(subreddit_list, posts_per_sub, classification_label):
    """Fetches posts from subreddits that are assumed to be 100% SFW or 100% NSFW."""
    all_posts = []
    for sub_name in subreddit_list:
        try:
            print(f"Fetching {posts_per_sub} posts from r/{sub_name}...")
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=posts_per_sub):
                all_posts.append([post.title, post.selftext, post.score, post.num_comments, sub_name, classification_label])
            time.sleep(1) # Be a good API citizen
        except Exception as e:
            print(f"⚠️ Could not fetch from r/{sub_name}. Error: {e}")
    return all_posts

def get_mixed_posts(subreddit_list, posts_to_scan):
    """
    Scans posts from mixed-content subreddits and classifies each post individually
    based on its 'over_18' (NSFW) flag provided by the Reddit API.
    """
    sfw_posts = []
    nsfw_posts = []
    for sub_name in subreddit_list:
        try:
            print(f"Scanning {posts_to_scan} posts from mixed-content subreddit r/{sub_name}...")
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=posts_to_scan):
                if post.over_18:
                    nsfw_posts.append([post.title, post.selftext, post.score, post.num_comments, sub_name, "NSFW"])
                else:
                    sfw_posts.append([post.title, post.selftext, post.score, post.num_comments, sub_name, "SFW"])
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ Could not fetch from r/{sub_name}. Error: {e}")
    return sfw_posts, nsfw_posts


try:
    # --- Step 1: Ingest from clearly SFW subreddits ---
    print("\n--- Ingesting SFW Posts ---")
    normal_posts = get_classified_posts(SFW_SUBREDDITS, POSTS_PER_SFW_SUB, "SFW")

    # --- Step 2: Ingest from clearly NSFW subreddits ---
    print("\n--- Ingesting NSFW Posts ---")
    anomaly_posts = get_classified_posts(NSFW_SUBREDDITS, POSTS_PER_NSFW_SUB, "NSFW")

    # --- Step 3: Scan and sort posts from mixed-content subreddits ---
    print("\n--- Scanning Mixed-Content Posts ---")
    mixed_sfw, mixed_nsfw = get_mixed_posts(MIXED_CONTENT_SUBREDDITS, POSTS_TO_SCAN_PER_MIXED_SUB)
    
    # Add the sorted posts to our main lists
    normal_posts.extend(mixed_sfw)
    anomaly_posts.extend(mixed_nsfw)

    # --- Combine, Clean, and Save Data ---
    total_normal = len(normal_posts)
    total_anomaly = len(anomaly_posts)
    
    if total_anomaly == 0:
        print("\n🚨 CRITICAL WARNING: No NSFW posts were found. The model cannot be trained correctly.")

    all_posts_data = normal_posts + anomaly_posts
    df = pd.DataFrame(all_posts_data, columns=["title", "body", "score", "num_comments", "subreddit", "classification"])
    
    # Clean the data by dropping posts with no text and shuffling
    df.dropna(subset=['title', 'body'], inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # --- Overwrite old data to ensure a fresh dataset for each run ---
    data_dir = "data"
    output_file = os.path.join(data_dir, "raw_posts.csv")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(output_file):
        print(f"\nFound old data file. Deleting {output_file} to create a fresh dataset.")
        os.remove(output_file)
    
    # Save the new, clean dataframe to a fresh CSV file
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*40)
    print("✅ Data Ingestion Complete!")
    print(f"Total SFW (Normal) Posts Fetched: {total_normal}")
    print(f"Total NSFW (Anomaly) Posts Fetched: {total_anomaly}")
    print(f"Total Usable Posts Saved to CSV: {len(df)}")
    print("="*40)

except Exception as e:
    print(f"❌ An unexpected error occurred during data ingestion: {e}")