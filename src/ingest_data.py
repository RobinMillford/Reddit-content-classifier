import praw
import pandas as pd
import os
from dotenv import load_dotenv
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import praw

load_dotenv()

# --- Multi-Label Classification Categories ---
CONTENT_CATEGORIES = {
    'safety': ['safe', 'nsfw'],
    'toxicity': ['non_toxic', 'toxic'],
    'sentiment': ['neutral', 'positive', 'negative'],
    'topic': ['technology', 'politics', 'entertainment', 'sports', 'health', 'education', 'business', 'relationships', 'adult_content', 'general'],
    'engagement': ['low_engagement', 'high_engagement']
}

# Keywords for topic classification (significantly enhanced)
TOPIC_KEYWORDS = {
    'technology': [
        # Core tech terms
        'tech', 'technology', 'software', 'hardware', 'computer', 'programming', 'app', 'application',
        'digital', 'internet', 'web', 'online', 'code', 'coding', 'development', 'developer',
        # AI/ML - Enhanced with LangChain, AI Agents, and AI community terms
        'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning', 'neural network',
        'algorithm', 'data science', 'automation', 'robot', 'robotics', 'chatgpt', 'gpt', 'openai',
        'langchain', 'ai agent', 'ai agents', 'autonomous agent', 'llm', 'large language model',
        'natural language processing', 'nlp', 'computer vision', 'reinforcement learning',
        'transformer', 'bert', 'gpt-3', 'gpt-4', 'claude', 'anthropic', 'hugging face',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
        'prompt engineering', 'fine-tuning', 'model training', 'inference', 'embeddings',
        'rag', 'retrieval augmented generation', 'vector database', 'semantic search',
        'ai safety', 'alignment', 'agi', 'artificial general intelligence', 'superintelligence',
        'machine reasoning', 'cognitive architecture', 'neural symbolic', 'multimodal ai',
        'ai ethics', 'bias', 'fairness', 'explainable ai', 'interpretability',
        'federated learning', 'transfer learning', 'few-shot learning', 'zero-shot learning',
        'generative ai', 'diffusion models', 'stable diffusion', 'dall-e', 'midjourney',
        'conversational ai', 'chatbot', 'virtual assistant', 'voice ai', 'speech recognition',
        'sentiment analysis', 'text classification', 'named entity recognition', 'text generation',
        'knowledge graph', 'ontology', 'semantic web', 'expert system', 'decision tree',
        'ensemble learning', 'gradient boosting', 'random forest', 'svm', 'support vector machine',
        'clustering', 'classification', 'regression', 'supervised learning', 'unsupervised learning',
        'semi-supervised learning', 'self-supervised learning', 'active learning',
        'data mining', 'big data', 'data pipeline', 'feature engineering', 'model deployment',
        'mlops', 'devops', 'continuous integration', 'model monitoring', 'a/b testing',
        # Platforms and tools
        'github', 'stackoverflow', 'python', 'javascript', 'react', 'nodejs', 'docker', 'kubernetes',
        'aws', 'cloud', 'database', 'sql', 'api', 'framework', 'library', 'linux', 'windows', 'mac',
        'azure', 'google cloud', 'gcp', 'serverless', 'microservices', 'containers',
        'jupyter', 'colab', 'anaconda', 'pip', 'conda', 'virtual environment',
        # Gaming tech (overlaps with entertainment but tech-focused)
        'pc gaming', 'console', 'gpu', 'cpu', 'ram', 'motherboard', 'overclocking', 'fps', 'graphics card',
        'nvidia', 'amd', 'intel', 'steam', 'epic games', 'origin', 'battlenet', 'gaming pc', 'rig',
        'ray tracing', 'dlss', 'fsr', 'vulkan', 'directx', 'opengl',
        # Mobile
        'android', 'ios', 'mobile', 'smartphone', 'tablet', 'iphone', 'samsung', 'google pixel',
        'flutter', 'react native', 'swift', 'kotlin', 'xamarin',
        # Emerging tech
        'blockchain', 'cryptocurrency', 'crypto', 'bitcoin', 'ethereum', 'nft', 'metaverse', 'vr', 'ar',
        'virtual reality', 'augmented reality', 'mixed reality', 'innovation', 'startup', 'cyber', 'cybersecurity', 
        'hacking', 'privacy', 'data breach', 'security', 'encryption', 'cryptography',
        'quantum computing', 'quantum', 'iot', 'internet of things', 'edge computing',
        '5g', '6g', 'wifi', 'bluetooth', 'nfc', 'rfid', 'sensors', 'embedded systems',
        'autonomous vehicles', 'self-driving', 'tesla', 'waymo', 'lidar', 'computer vision',
        'drone', 'uav', 'robotics', 'automation', 'industrial automation', 'smart home',
        'smart city', 'digital twin', 'simulation', 'modeling', 'optimization'
    ],
    'politics': [
        # Elections and voting
        'election', 'vote', 'voting', 'ballot', 'poll', 'campaign', 'candidate', 'primary',
        # Government
        'government', 'president', 'congress', 'senate', 'house', 'parliament', 'minister',
        'political', 'policy', 'politics', 'democrat', 'republican', 'conservative', 'liberal',
        'law', 'legislation', 'regulation', 'bill', 'act', 'amendment', 'constitution',
        # International
        'international', 'diplomacy', 'embassy', 'foreign policy', 'trade war', 'sanctions',
        'nato', 'un', 'united nations', 'eu', 'european union',
        # Issues
        'immigration', 'healthcare', 'climate change', 'abortion', 'gun control', 'taxation',
        'welfare', 'social security', 'unemployment', 'inflation'
    ],
    'entertainment': [
        # Movies and TV
        'movie', 'film', 'cinema', 'hollywood', 'tv', 'television', 'show', 'series', 'episode',
        'streaming', 'netflix', 'disney', 'disney plus', 'hulu', 'amazon prime', 'hbo', 'hbo max',
        'documentary', 'actor', 'actress', 'director', 'producer', 'screenplay', 'trailer', 'premiere',
        'marvel', 'dc', 'star wars', 'mcu', 'superhero', 'anime', 'manga', 'cartoon', 'animation',
        # Music
        'music', 'song', 'album', 'artist', 'band', 'singer', 'concert', 'tour', 'festival',
        'spotify', 'youtube', 'soundcloud', 'playlist', 'lyrics', 'genre', 'pop', 'rock', 'rap',
        'hip hop', 'country', 'jazz', 'classical', 'electronic', 'edm', 'kpop', 'taylor swift',
        # Gaming entertainment
        'gaming', 'game', 'video game', 'esports', 'twitch', 'streamer', 'lets play', 'speedrun',
        'minecraft', 'fortnite', 'league of legends', 'overwatch', 'call of duty', 'gta', 'pokemon',
        'nintendo', 'playstation', 'xbox', 'steam', 'indie game', 'mmorpg', 'fps', 'rpg', 'moba',
        # General entertainment
        'celebrity', 'fame', 'award', 'oscar', 'emmy', 'grammy', 'golden globe', 'red carpet', 'gossip',
        'entertainment', 'media', 'news', 'tabloid', 'paparazzi', 'viral', 'meme', 'tiktok', 'instagram'
    ],
    'sports': [
        # Major sports
        'football', 'nfl', 'basketball', 'nba', 'baseball', 'mlb', 'soccer', 'fifa', 'world cup',
        'hockey', 'nhl', 'tennis', 'golf', 'boxing', 'mma', 'ufc', 'wrestling', 'wwe',
        # Olympics and competitions
        'olympics', 'championship', 'tournament', 'playoff', 'finals', 'league', 'season',
        'player', 'team', 'coach', 'athlete', 'sport', 'match', 'game', 'score', 'goal',
        'touchdown', 'homerun', 'knockout', 'victory', 'defeat', 'win', 'loss',
        # College sports
        'college football', 'march madness', 'ncaa', 'draft', 'rookie', 'veteran',
        # Fantasy and betting
        'fantasy football', 'fantasy basketball', 'betting', 'odds', 'sportsbook'
    ],
    'health': [
        # Medical
        'health', 'medical', 'medicine', 'doctor', 'physician', 'nurse', 'hospital', 'clinic',
        'patient', 'treatment', 'therapy', 'medication', 'prescription', 'surgery', 'operation',
        'diagnosis', 'symptoms', 'disease', 'illness', 'infection', 'virus', 'bacteria',
        # Mental health
        'mental health', 'depression', 'anxiety', 'stress', 'therapy', 'counseling', 'psychiatrist',
        'psychology', 'ptsd', 'bipolar', 'schizophrenia', 'autism', 'adhd',
        # Fitness and wellness
        'fitness', 'exercise', 'workout', 'gym', 'weight loss', 'diet', 'nutrition', 'protein',
        'calories', 'muscle', 'strength', 'cardio', 'running', 'yoga', 'meditation',
        'wellness', 'healthy', 'organic', 'supplement', 'vitamin'
    ],
    'education': [
        # Institutions
        'school', 'university', 'college', 'academy', 'institute', 'campus', 'classroom',
        'education', 'academic', 'student', 'teacher', 'professor', 'instructor', 'tutor',
        # Academic activities
        'learning', 'study', 'studying', 'homework', 'assignment', 'project', 'research',
        'thesis', 'dissertation', 'paper', 'essay', 'exam', 'test', 'quiz', 'grade',
        'degree', 'bachelor', 'master', 'phd', 'doctorate', 'graduation', 'diploma',
        # Subjects
        'course', 'class', 'subject', 'math', 'science', 'history', 'english', 'literature',
        'chemistry', 'physics', 'biology', 'economics', 'psychology', 'sociology',
        # Online learning
        'online learning', 'e-learning', 'mooc', 'coursera', 'udemy', 'khan academy'
    ],
    'business': [
        # Finance
        'business', 'finance', 'financial', 'money', 'investment', 'investing', 'stock', 'stocks',
        'market', 'wall street', 'trading', 'trader', 'portfolio', 'dividend', 'profit', 'revenue',
        'earnings', 'loss', 'debt', 'credit', 'loan', 'mortgage', 'insurance',
        # Economy
        'economy', 'economic', 'gdp', 'inflation', 'recession', 'unemployment', 'federal reserve',
        'interest rate', 'tax', 'taxation', 'budget', 'fiscal', 'monetary',
        # Corporate
        'company', 'corporation', 'startup', 'entrepreneur', 'ceo', 'cfo', 'management',
        'board', 'shareholder', 'merger', 'acquisition', 'ipo', 'valuation',
        # Work
        'job', 'career', 'employment', 'salary', 'wage', 'benefits', 'retirement', 'pension',
        'workplace', 'office', 'remote work', 'freelance', 'gig economy'
    ],
    'relationships': [
        # Romantic relationships
        'relationship', 'dating', 'love', 'romance', 'romantic', 'couple', 'partner',
        'boyfriend', 'girlfriend', 'husband', 'wife', 'marriage', 'married', 'wedding',
        'engagement', 'engaged', 'breakup', 'divorce', 'separation', 'cheating', 'affair',
        'tinder', 'bumble', 'dating app', 'online dating', 'long distance',
        # Family
        'family', 'mom', 'mother', 'dad', 'father', 'parent', 'child', 'kids', 'baby',
        'son', 'daughter', 'sister', 'brother', 'sibling', 'grandparent', 'grandmother',
        'grandfather', 'aunt', 'uncle', 'cousin', 'nephew', 'niece',
        # Social
        'friend', 'friendship', 'best friend', 'social', 'party', 'hangout', 'meetup'
    ],
    'adult_content': [
        # Explicit sexual terms
        'sex', 'sexual', 'fucking', 'fuck', 'cock', 'dick', 'penis', 'pussy', 'vagina',
        'cum', 'cumming', 'orgasm', 'climax', 'masturbate', 'masturbation', 'jerk off',
        'blow job', 'blowjob', 'oral sex', 'anal sex', 'penetration', 'thrust',
        # Body parts
        'nipples', 'breasts', 'tits', 'boobs', 'ass', 'butt', 'butthole', 'anus',
        'clitoris', 'clit', 'labia', 'testicles', 'balls', 'shaft', 'head',
        # Adult industry
        'porn', 'pornography', 'adult film', 'adult video', 'xxx', 'nsfw', 'adult',
        'erotic', 'erotica', 'nude', 'naked', 'strip', 'stripper', 'escort', 'prostitute',
        # Activities
        'intimate', 'intimacy', 'foreplay', 'fetish', 'kink', 'bdsm', 'bondage',
        'dominance', 'submission', 'roleplay', 'fantasy', 'threesome', 'orgy',
        # Adjectives
        'sexy', 'hot', 'horny', 'aroused', 'turned on', 'wet', 'hard', 'stiff'
    ]
}

# Enhanced toxicity keywords with context awareness and broader coverage
TOXIC_KEYWORDS = {
    'hate_speech': [
        'hate', 'racist', 'racism', 'nazi', 'fascist', 'supremacist', 'bigot', 'xenophobic',
        'antisemitic', 'islamophobic', 'kill yourself', 'kys', 'die', 'murder', 'lynch',
        'ethnic cleansing', 'genocide', 'inferior race', 'subhuman', 'savage'
    ],
    'harassment': [
        'harassment', 'harass', 'stalking', 'stalk', 'doxxing', 'dox', 'threatening', 'threat',
        'intimidation', 'intimidate', 'cyberbully', 'bullying', 'troll', 'trolling',
        'witch hunt', 'mob', 'cancel', 'brigade', 'spam', 'abuse', 'victim blaming'
    ],
    'insults': [
        'stupid', 'idiot', 'moron', 'retard', 'retarded', 'dumb', 'dumbass', 'fool', 'foolish',
        'bitch', 'asshole', 'piece of shit', 'scumbag', 'loser', 'pathetic', 'worthless',
        'trash', 'garbage', 'waste of space', 'fat', 'ugly', 'disgusting', 'pig'
    ],
    'threats': [
        'threat', 'threaten', 'kill', 'murder', 'hurt', 'harm', 'violence', 'violent',
        'assault', 'attack', 'beat up', 'punch', 'shoot', 'stab', 'bomb', 'terrorism',
        'terrorist', 'explosive', 'weapon', 'gun', 'knife', 'poison'
    ],
    'discrimination': [
        'sexist', 'sexism', 'misogyny', 'misogynist', 'homophobic', 'homophobia',
        'transphobic', 'transphobia', 'discriminate', 'discrimination', 'prejudice',
        'stereotyping', 'profiling', 'ageism', 'ableism', 'classism'
    ],
    'self_harm': [
        'suicide', 'suicidal', 'self harm', 'cutting', 'overdose', 'jumping off',
        'hanging', 'kill myself', 'end it all', 'want to die', 'better off dead'
    ]
}

# Adult content indicators (not necessarily toxic) - expanded
ADULT_CONTENT_KEYWORDS = [
    # Explicit sexual terms
    'sex', 'sexual', 'fuck', 'fucking', 'cock', 'dick', 'penis', 'pussy', 'vagina',
    'cum', 'cumming', 'orgasm', 'climax', 'masturbate', 'masturbation', 'jerk off',
    'blow job', 'blowjob', 'handjob', 'oral sex', 'anal sex', 'penetration',
    # Body parts
    'nipples', 'breasts', 'tits', 'boobs', 'ass', 'butt', 'butthole', 'anus',
    'clitoris', 'clit', 'labia', 'testicles', 'balls', 'shaft', 'head',
    # Adult industry and content
    'porn', 'pornography', 'xxx', 'nsfw', 'adult', 'erotic', 'erotica',
    'nude', 'naked', 'strip', 'stripper', 'escort', 'prostitute', 'hooker',
    # Activities and behaviors
    'intimate', 'intimacy', 'foreplay', 'fetish', 'kink', 'bdsm', 'bondage',
    'dominance', 'submission', 'roleplay', 'fantasy', 'threesome', 'orgy',
    'gangbang', 'bukkake', 'creampie', 'deepthroat', 'facefuck',
    # Adjectives and states
    'sexy', 'hot', 'horny', 'aroused', 'turned on', 'wet', 'hard', 'stiff',
    'naughty', 'dirty', 'slutty', 'kinky', 'perverted', 'lustful'
]

# Positive emotional indicators - expanded for diverse content
POSITIVE_INDICATORS = [
    # General positive
    'amazing', 'awesome', 'fantastic', 'wonderful', 'excellent', 'great', 'good', 
    'beautiful', 'love', 'happy', 'joy', 'joyful', 'excited', 'thrilled', 'pleased',
    'satisfied', 'grateful', 'blessed', 'lucky', 'perfect', 'incredible', 'outstanding',
    # Entertainment/gaming positive
    'epic', 'legendary', 'masterpiece', 'brilliant', 'genius', 'hilarious', 'funny',
    'entertaining', 'addictive', 'engaging', 'immersive', 'captivating',
    # Success/achievement
    'success', 'successful', 'achievement', 'victory', 'win', 'winner', 'champion',
    'accomplished', 'proud', 'milestone', 'breakthrough', 'triumph',
    # Quality indicators
    'high quality', 'premium', 'top tier', 'first class', 'stellar', 'superb',
    'magnificent', 'phenomenal', 'exceptional', 'remarkable', 'impressive',
    # Social positive
    'supportive', 'helpful', 'kind', 'generous', 'caring', 'thoughtful', 'sweet',
    'heartwarming', 'inspiring', 'motivating', 'uplifting', 'encouraging'
]

# Negative emotional indicators - expanded for diverse content  
NEGATIVE_INDICATORS = [
    # General negative
    'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'angry', 'furious',
    'disgusting', 'disappointing', 'frustrated', 'annoying', 'pathetic',
    'useless', 'worthless', 'depressing', 'sad', 'miserable', 'upset', 'unfortunate',
    # Intensity negative
    'disaster', 'catastrophe', 'nightmare', 'hellish', 'toxic', 'poisonous',
    'cancerous', 'plague', 'virus', 'disease', 'infection', 'contaminated',
    # Failure/disappointment
    'failure', 'failed', 'broken', 'ruined', 'destroyed', 'wasted', 'lost',
    'defeat', 'beaten', 'crushed', 'devastated', 'shattered', 'disappointed',
    # Quality negative
    'trash', 'garbage', 'junk', 'crap', 'shit', 'sucks', 'boring', 'dull',
    'bland', 'mediocre', 'subpar', 'inferior', 'cheap', 'fake', 'knockoff',
    # Emotional states
    'depressed', 'anxious', 'stressed', 'overwhelmed', 'exhausted', 'tired',
    'sick', 'ill', 'hurt', 'pain', 'suffering', 'agony', 'torture', 'nightmare'
]

# Engagement indicators - significantly expanded
HIGH_ENGAGEMENT_KEYWORDS = [
    # Excitement/reaction
    'wow', 'omg', 'amazing', 'incredible', 'unbelievable', 'shocking', 'epic',
    'insane', 'crazy', 'wild', 'nuts', 'mind-blowing', 'jaw-dropping', 'stunning',
    'breathtaking', 'spectacular', 'phenomenal', 'extraordinary', 'remarkable',
    # Viral/trending
    'viral', 'trending', 'popular', 'hot', 'trending now', 'going viral',
    'blown up', 'exploded', 'massive', 'huge', 'enormous',
    # Call to action
    'must see', 'check this out', 'look at this', 'watch this', 'see this',
    'share this', 'repost', 'spread the word', 'tell everyone', 'upvote',
    # Breaking news/updates
    'breaking', 'urgent', 'alert', 'news flash', 'just in', 'developing',
    'update:', 'edit:', 'new:', 'latest:', 'fresh:', 'exclusive',
    # Superlatives
    'best ever', 'worst ever', 'greatest', 'biggest', 'smallest', 'fastest',
    'strongest', 'weirdest', 'strangest', 'funniest', 'scariest',
    # Gaming/entertainment specific
    'legendary', 'godlike', 'op', 'overpowered', 'clutch', 'pro', 'noob',
    'fail', 'epic fail', 'win', 'epic win', 'pwned', 'rekt', 'destroyed',
    # Social media specific
    'lol', 'lmao', 'rofl', 'lmfao', 'wtf', 'omfg', 'fml', 'smh', 'imo', 'tbh',
    # Punctuation indicators (handled separately in function)
    '!!!', '???', '!?!', '!!!!'  # These are counted differently
]

# --- Configuration ---
# Balanced configuration for optimal multi-label training
# Equal posts per SFW/NSFW for better binary classification balance
# Higher mixed content scanning for diverse multi-label examples
POSTS_PER_SFW_SUB = 180
POSTS_PER_NSFW_SUB = 180  
POSTS_TO_SCAN_PER_MIXED_SUB = 250

# --- Get Credentials ---
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

if not all([client_id, client_secret, user_agent]):
    raise ValueError("Reddit credentials not set in .env file.")

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

def classify_topic(text, subreddit=None):
    """Enhanced topic classification with subreddit context and better scoring."""
    text_lower = text.lower()
    topic_scores = {}
    
    # Use subreddit as strong context signal
    if subreddit:
        subreddit_lower = subreddit.lower()
        
        # Direct subreddit mapping based on your subreddit lists
        tech_subs = ['technology', 'programming', 'pcmasterrace', 'buildapc', 'learnprogramming', 
                     'compsci', 'hardware', 'pcgaming', 'apple', 'iphone', 'android', 'cryptocurrency', 
                     'bitcoin', 'ethereum', 'singularity', 'futurology', 'gadgets']
        gaming_subs = ['gaming', 'games', 'minecraft', 'overwatch', 'fortnite', 'leagueoflegends', 
                       'ps4', 'ps5', 'xboxone', 'xboxseriesx', 'nintendo', 'nintendoswitch', 
                       'steam', 'genshin_impact', 'destiny', 'eldenring', 'gta']
        sports_subs = ['sports', 'nba', 'nfl', 'soccer', 'football', 'cfb', 'formula1', 
                       'premierleague', 'mma', 'fantasyfootball']
        politics_subs = ['politics', 'worldnews', 'news', 'worldpolitics', 'politicalhumor']
        health_subs = ['fitness', 'loseit', 'keto', 'running', 'bodyweightfitness', 
                       'meditation', 'yoga', 'mentalhealth', 'health']
        business_subs = ['personalfinance', 'entrepreneur', 'stocks', 'investing', 
                         'stockmarket', 'daytrading', 'economics', 'careerguidance']
        
        # Strong subreddit signals
        if any(sub in subreddit_lower for sub in tech_subs):
            topic_scores['technology'] = topic_scores.get('technology', 0) + 10
        if any(sub in subreddit_lower for sub in gaming_subs):
            topic_scores['technology'] = topic_scores.get('technology', 0) + 8  # Gaming is tech-related
        if any(sub in subreddit_lower for sub in sports_subs):
            topic_scores['sports'] = topic_scores.get('sports', 0) + 10
        if any(sub in subreddit_lower for sub in politics_subs):
            topic_scores['politics'] = topic_scores.get('politics', 0) + 10
        if any(sub in subreddit_lower for sub in health_subs):
            topic_scores['health'] = topic_scores.get('health', 0) + 10
        if any(sub in subreddit_lower for sub in business_subs):
            topic_scores['business'] = topic_scores.get('business', 0) + 10
    
    # Check for adult content first (higher threshold)
    adult_score = sum(1 for keyword in ADULT_CONTENT_KEYWORDS if keyword in text_lower)
    if adult_score >= 4:  # Increased threshold for better precision
        topic_scores['adult_content'] = adult_score * 2
    
    # Score all topics based on text content
    for topic, keywords in TOPIC_KEYWORDS.items():
        if topic not in topic_scores:
            topic_scores[topic] = 0
        
        # Count keyword matches with weighted scoring
        for keyword in keywords:
            if keyword in text_lower:
                # Longer keywords get higher weight
                weight = max(1, len(keyword.split()))
                topic_scores[topic] += weight
    
    # Relationship content detection
    relationship_indicators = ['relationship', 'dating', 'boyfriend', 'girlfriend', 'marriage', 
                              'family', 'mom', 'dad', 'aita', 'relationship_advice']
    if any(indicator in text_lower for indicator in relationship_indicators):
        topic_scores['relationships'] = topic_scores.get('relationships', 0) + 5
    
    # Return the topic with highest score, with minimum threshold
    if topic_scores and max(topic_scores.values()) >= 2:
        return max(topic_scores, key=topic_scores.get)
    return 'general'

def classify_toxicity(text, subreddit=None):
    """Enhanced toxicity classification with subreddit context and severity levels."""
    text_lower = text.lower()
    
    # Count different types of toxic content with severity weighting
    toxicity_score = 0
    
    for category, keywords in TOXIC_KEYWORDS.items():
        category_count = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Weight different categories by severity
        if category in ['hate_speech', 'threats', 'self_harm']:
            toxicity_score += category_count * 5  # Most severe
        elif category in ['harassment', 'discrimination']:
            toxicity_score += category_count * 3  # Moderately severe
        else:
            toxicity_score += category_count * 2  # Less severe but still toxic
    
    # Context from subreddit - some subs are more tolerant of strong language
    if subreddit:
        subreddit_lower = subreddit.lower()
        # Subreddits where strong language is more common but not necessarily toxic
        casual_subs = ['wallstreetbets', 'roastme', 'circlejerk', 'dankmemes', 
                       'okbuddyretard', 'shitposting', 'comedyheaven']
        if any(sub in subreddit_lower for sub in casual_subs):
            toxicity_score = max(0, toxicity_score - 2)  # Reduce toxicity score
    
    # Adult content alone is not toxic unless combined with harmful elements
    adult_content_count = sum(1 for keyword in ADULT_CONTENT_KEYWORDS if keyword in text_lower)
    
    # Only consider toxic if there are actual harmful keywords, not just adult content
    # Adjusted threshold for better precision
    if toxicity_score >= 3:
        return 'toxic'
    elif adult_content_count > 0 and toxicity_score <= 1:
        return 'non_toxic'  # Adult but not harmful
    else:
        return 'non_toxic'

def classify_sentiment(text, subreddit=None):
    """Enhanced sentiment analysis with subreddit context and better scoring."""
    text_lower = text.lower()
    
    # Count positive and negative indicators with weighted scoring
    positive_score = 0
    negative_score = 0
    
    # Weight longer phrases more heavily
    for word in POSITIVE_INDICATORS:
        if word in text_lower:
            weight = max(1, len(word.split()))
            positive_score += weight
    
    for word in NEGATIVE_INDICATORS:
        if word in text_lower:
            weight = max(1, len(word.split()))
            negative_score += weight
    
    # Look for enthusiasm markers with more sophisticated detection
    enthusiasm_score = 0
    enthusiasm_score += text.count('!') * 0.5  # Exclamation marks
    enthusiasm_score += text.count('?') * 0.2  # Questions can indicate engagement
    enthusiasm_score += len([word for word in text.split() if word.isupper() and len(word) > 2]) * 1.5  # ALL CAPS
    
    # Detect repeated characters (enthusiasm indicators)
    import re
    repeated_chars = len(re.findall(r'(.)\1{2,}', text_lower))  # 3+ repeated chars
    enthusiasm_score += repeated_chars * 0.5
    
    # Subreddit context for sentiment
    if subreddit:
        subreddit_lower = subreddit.lower()
        
        # Positive-leaning subreddits
        positive_subs = ['wholesomememes', 'mademesmile', 'upliftingnews', 'getmotivated', 
                        'humansbeingbros', 'aww', 'eyebleach', 'contagiouslaughter']
        # Negative-leaning subreddits
        negative_subs = ['tifu', 'cringe', 'facepalm', 'mildlyinfuriating', 'iamatotalpieceofshit',
                        'trashy', 'insanepeoplefacebook', 'leopardsatemyface']
        
        if any(sub in subreddit_lower for sub in positive_subs):
            positive_score += 2
        elif any(sub in subreddit_lower for sub in negative_subs):
            negative_score += 1  # Smaller boost for negative
    
    # Adult content sentiment analysis
    adult_content_count = sum(1 for keyword in ADULT_CONTENT_KEYWORDS if keyword in text_lower)
    if adult_content_count > 0:
        # Adult content can be positive if expressed enthusiastically
        if enthusiasm_score > 3:
            positive_score += 3
        elif enthusiasm_score < 1 and negative_score > 0:
            negative_score += 1  # Sad/negative adult content
    
    # Calculate final sentiment with adjusted thresholds
    total_positive = positive_score + enthusiasm_score
    total_negative = negative_score
    
    # More nuanced sentiment determination
    if total_positive > total_negative + 2:  # Clear positive signal needed
        return 'positive'
    elif total_negative > total_positive + 1:  # Negative threshold
        return 'negative'
    else:
        return 'neutral'

def classify_engagement(score, num_comments):
    """Enhanced engagement classification with text-based indicators."""
    # Base engagement from numbers
    engagement_score = score + (num_comments * 2)
    
    return 'high_engagement' if engagement_score > 30 else 'low_engagement'  # Lowered threshold

def classify_engagement_with_text(text, score, num_comments, subreddit=None):
    """Enhanced engagement classification with text analysis and subreddit context."""
    text_lower = text.lower()
    
    # Base score from Reddit metrics (normalized)
    base_engagement = min(50, score + (num_comments * 2))  # Cap to prevent dominance
    
    # Text-based engagement indicators
    text_engagement = 0
    
    # Punctuation enthusiasm
    text_engagement += min(20, text.count('!') * 2)  # Cap exclamation bonus
    text_engagement += min(10, text.count('?') * 1)  # Questions engage readers
    
    # ALL CAPS detection (enthusiasm)
    caps_words = [word for word in text.split() if word.isupper() and len(word) > 2]
    text_engagement += min(15, len(caps_words) * 3)
    
    # High engagement keywords
    engagement_keywords_count = sum(1 for keyword in HIGH_ENGAGEMENT_KEYWORDS if keyword in text_lower)
    text_engagement += min(25, engagement_keywords_count * 4)
    
    # Length-based engagement (longer posts often get more engagement)
    text_length = len(text)
    if text_length > 1000:
        text_engagement += 15
    elif text_length > 500:
        text_engagement += 10
    elif text_length > 200:
        text_engagement += 5
    
    # Special patterns that drive engagement
    import re
    # Questions drive engagement
    question_count = len(re.findall(r'\?', text))
    text_engagement += min(10, question_count * 2)
    
    # Numbered lists/steps (highly engaging)
    numbered_lists = len(re.findall(r'\d+\.\s', text))
    text_engagement += min(15, numbered_lists * 3)
    
    # URLs/links (shareable content)
    urls = len(re.findall(r'http[s]?://|www\.', text_lower))
    text_engagement += min(10, urls * 5)
    
    # Subreddit context for engagement
    subreddit_engagement = 0
    if subreddit:
        subreddit_lower = subreddit.lower()
        
        # High-engagement subreddits
        viral_subs = ['askreddit', 'tifu', 'aita', 'relationship_advice', 'nosleep',
                     'wallstreetbets', 'memes', 'dankmemes', 'publicfreakout']
        # Niche/specialized subs (lower baseline engagement)
        niche_subs = ['programming', 'compsci', 'philosophy', 'economics', 'science']
        
        if any(sub in subreddit_lower for sub in viral_subs):
            subreddit_engagement += 10
        elif any(sub in subreddit_lower for sub in niche_subs):
            subreddit_engagement -= 5  # Lower threshold for these subs
    
    # Calculate total engagement score
    total_engagement = base_engagement + text_engagement + subreddit_engagement
    
    # Dynamic threshold based on content type
    threshold = 35
    if subreddit and any(sub in subreddit.lower() for sub in ['programming', 'science', 'philosophy']):
        threshold = 25  # Lower threshold for academic/technical content
    
    return 'high_engagement' if total_engagement > threshold else 'low_engagement'

def get_multi_label_classification(title, body, score, num_comments, subreddit, is_nsfw):
    """Enhanced multi-label classification with comprehensive context awareness."""
    full_text = f"{title} {body if body else ''}"
    
    # Use subreddit context in all classifications
    return {
        'safety': 'nsfw' if is_nsfw else 'safe',
        'toxicity': classify_toxicity(full_text, subreddit),
        'sentiment': classify_sentiment(full_text, subreddit),
        'topic': classify_topic(full_text, subreddit),
        'engagement': classify_engagement_with_text(full_text, score, num_comments, subreddit)
    }

def get_classified_posts(subreddit_list, posts_per_sub, classification_label):
    """Fetches posts from subreddits that are assumed to be 100% SFW or 100% NSFW."""
    all_posts = []
    for sub_name in subreddit_list:
        try:
            print(f"Fetching {posts_per_sub} posts from r/{sub_name}...")
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=posts_per_sub):
                # Get multi-label classifications
                labels = get_multi_label_classification(
                    post.title, 
                    post.selftext, 
                    post.score, 
                    post.num_comments, 
                    sub_name, 
                    classification_label == 'NSFW'
                )
                
                # Create post data with all labels
                post_data = [
                    post.title, 
                    post.selftext, 
                    post.score, 
                    post.num_comments, 
                    sub_name, 
                    classification_label,  # Keep original for compatibility
                    labels['safety'],
                    labels['toxicity'],
                    labels['sentiment'],
                    labels['topic'],
                    labels['engagement']
                ]
                all_posts.append(post_data)
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ Could not fetch from r/{sub_name}. Error: {e}")
    return all_posts

def get_mixed_posts(subreddit_list, posts_to_scan):
    """
    Scans posts from mixed-content subreddits and classifies each post individually
    based on its 'over_18' (NSFW) flag and other characteristics.
    """
    sfw_posts = []
    nsfw_posts = []
    for sub_name in subreddit_list:
        try:
            print(f"Scanning {posts_to_scan} posts from mixed-content subreddit r/{sub_name}...")
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=posts_to_scan):
                # Get multi-label classifications
                labels = get_multi_label_classification(
                    post.title, 
                    post.selftext, 
                    post.score, 
                    post.num_comments, 
                    sub_name, 
                    post.over_18
                )
                
                # Create post data with all labels
                post_data = [
                    post.title, 
                    post.selftext, 
                    post.score, 
                    post.num_comments, 
                    sub_name, 
                    "NSFW" if post.over_18 else "SFW",  # Keep original for compatibility
                    labels['safety'],
                    labels['toxicity'],
                    labels['sentiment'],
                    labels['topic'],
                    labels['engagement']
                ]
                
                if post.over_18:
                    nsfw_posts.append(post_data)
                else:
                    sfw_posts.append(post_data)
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ Could not fetch from r/{sub_name}. Error: {e}")
    return sfw_posts, nsfw_posts

# Using set() to automatically remove any duplicates, then converting back to a list.
SFW_SUBREDDITS = list(set([
    "announcements", "funny", "aww", "Music", "memes", "movies", "Showerthoughts", "science", "pics", "Jokes", "news", "space", 
    "videos", "DIY", "books", "askscience", "nottheonion", "mildlyinteresting", "food", "GetMotivated", "EarthPorn", 
    "explainlikeimfive", "LifeProTips", "gadgets", "IAmA", "Art", "sports", "dataisbeautiful", "gifs", "Futurology", 
    "personalfinance", "photoshopbattles", "Documentaries", "UpliftingNews", "Damnthatsinteresting", "WritingPrompts", 
    "blog", "OldSchoolCool", "technology", "history", "philosophy", "wholesomememes", "listentothis", "television", 
    "InternetIsBeautiful", "NatureIsFuckingLit", "nba", "pcmasterrace", "lifehacks", "interestingasfuck", "TwoXChromosomes", 
    "travel", "anime", "ContagiousLaughter", "HistoryMemes", "Fitness", "nfl", "dadjokes", "oddlysatisfying", "NetflixBestOf", 
    "Unexpected", "MadeMeSmile", "EatCheapAndHealthy", "tattoos", "AdviceAnimals", "mildlyinfuriating", "CryptoCurrency", 
    "ChatGPT", "europe", "nextfuckinglevel", "BeAmazed", "FoodPorn", "stocks", "soccer", "Minecraft", "AnimalsBeingDerps", 
    "FunnyAnimals", "facepalm", "Parenting", "leagueoflegends", "cats", "rarepuppers", "PS5", "WatchPeopleDieInside", 
    "gardening", "buildapc", "Bitcoin", "NintendoSwitch", "me_irl", "Whatcouldgowrong", "place", "itookapicture", "cars", 
    "therewasanattempt", "CozyPlaces", "MakeupAddiction", "AnimalsBeingBros", "programming", "HumansBeingBros", "AskMen", 
    "blursedimages", "AnimalsBeingJerks", "Frugal", "starterpacks", "socialskills", "apple", "dating", "malefashionadvice", 
    "nevertellmetheodds", "BlackPeopleTwitter", "Overwatch", "Awwducational", "coolguides", "entertainment", "woodworking", 
    "dankmemes", "CrappyDesign", "nutrition", "RoastMe", "nasa", "femalefashionadvice", "NoStupidQuestions", "technicallythetruth", 
    "foodhacks", "MapPorn", "drawing", "TravelHacks", "MealPrepSunday", "FortNiteBR", "AskWomen", "PS4", "anime_irl", "Sneakers", 
    "photography", "YouShouldKnow", "Economics", "biology", "ModernWarfareII", "pokemongo", "backpacking", "XboxSeriesX", "boardgames", 
    "bestof", "formula1", "battlestations", "camping", "Shoestring", "OnePiece", "unitedkingdom", "hiphopheads", "popculturechat", 
    "streetwear", "Outdoors", "trippinthroughtime", "Survival", "PremierLeague", "strength_training", "Fauxmoi", "Daytrading", "Steam", 
    "KidsAreFuckingStupid", "SkincareAddiction", "psychology", "BikiniBottomTwitter", "Cooking", "manga", "pettyrevenge", "slowcooking", 
    "Entrepreneur", "careerguidance", "pokemon", "marvelstudios", "dating_advice", "instant_regret", "homeautomation", "Eyebleach", 
    "HomeImprovement", "UnresolvedMysteries", "solotravel", "Hair", "ProgrammerHumor", "Design", "bodyweightfitness", "scifi", "blackmagicfuckery", 
    "hardware", "CFB", "marvelmemes", "painting", "Eldenring", "ExpectationVsReality", "nonononoyes", "woahdude", "learnprogramming", "MaliciousCompliance", 
    "iphone", "MovieDetails", "DnD", "StarWars", "MyPeopleNeedMe", "Animemes", "roadtrip", "comicbooks", "IdiotsInCars", "loseit", "ThriftStoreHauls", "running", 
    "compsci", "spaceporn", "Wellthatsucks", "motorcycles", "xboxone", "HighQualityGifs", "keto", "LivestreamFail", "standupshots", "Nails", "productivity", "math", 
    "Baking", "reactiongifs", "podcasts", "AITAH", "15minutefood", "sciencememes", "pcgaming", "chemistry", "WeAreTheMusicMakers", "changemyview", "Fantasy", "PrequelMemes", 
    "RelationshipMemes", "kpop", "canada", "DigitalPainting", "oddlyspecific", "DiWHY", "maybemaybemaybe", "spacex", "ethereum", "TaylorSwift", "singularity", "Health", "MMA", 
    "relationships", "Genshin_Impact", "OutOfTheLoop", "indieheads", "gameofthrones", "StockMarket", "HealthyFood", "ArtPorn", "Meditation", "DunderMifflin", "recipes", "google", 
    "PewdiepieSubmissions", "HolUp", "homestead", "teslamotors", "JapanTravel", "OUTFITS", "MinecraftMemes", "Games", "UFOs", "GTA", "howto", "DestinyTheGame", "HistoryPorn", 
    "PeopleFuckingDying", "fantasyfootball", "yoga", "cursedcomments", "GifRecipes", "MurderedByWords", "harrypotter", "zelda", "Marvel", "raspberry_pi", "Perfectfit",
    "LangChain", "AI_Agents", "ArtificialInteligence"
]))

# Subreddits that are almost exclusively NSFW
NSFW_SUBREDDITS = list(set([
    "gonewild", "nsfw", "RealGirls", "porn", "hentai", "cumsluts", "rule34", "LegalTeens", "collegesluts", "GirlsFinishingTheJob", 
    "AsiansGoneWild", "NSFW_GIF", "onlyfansgirls101", "pussy", "nsfwhardcore", "BreedingMaterial", "TittyDrop", "PetiteGoneWild", 
    "milf", "Nude_Selfie", "BustyPetite", "Slut", "boobs", "ass", "Nudes", "latinas", "tiktokporn", "adorableporn", "OnlyFans101", 
    "needysluts", "tiktoknsfw", "GodPussy", "Blowjobs", "celebnsfw", "AsianHotties", "bigasses", "pawg", "nsfw_gifs", "barelylegalteens", 
    "anal", "LipsThatGrip", "porninfifteenseconds", "pornID", "GOONED", "holdthemoan", "gothsluts", "xsmallgirls", "BiggerThanYouThought", 
    "SluttyConfessions", "lesbians", "DadWouldBeProud", "juicyasians", "curvy", "squirting", "chubby", "JizzedToThis", "OnOff", "SheLikesItRough", 
    "SheFucksHim", "ThickThighs", "gonewildaudio", "freeuse", "HENTAI_GIF", "deepthroat", "18_19", "Gonewild18", "Cuckold", "tiktokthots", "BigBoobsGW", 
    "FemBoys", "fuckdoll", "cumfetish", "dirtyr4r", "Hotwife", "cosplaygirls", "naturaltitties", "bigtiddygothgf", "asstastic", "girlsmasturbating", 
    "creampies", "grool", "thickwhitegirls", "asshole", "HugeDickTinyChick", "Amateur", "RealHomePorn", "traps", "Stacked", "thick", "TeenBeauties", 
    "slutsofsnapchat", "Upskirt", "BlowJob", "nsfwcosplay", "PublicFlashing", "WatchItForThePlot", "FitNakedGirls", "dirtysmall", "workgonewild", 
    "blackchickswhitedicks", "paag", "TooCuteForPorn", "HappyEmbarrassedGirls", "amateurcumsluts", "HomemadeNsfw", "altgonewild", "Pussy_Perfection", 
    "gettingherselfoff", "palegirls", "Threesome", "NaughtyWives", "AnalGW", "SmallCutie", "public", "gonewild30plus", "OnlyFansPromotions", 
    "PublicSexPorn", "EGirls", "ButtsAndBareFeet", "rapefantasies", "booty_queens", "WomenBendingOver", "NSFW411", "YoungGirlsGoneWild", 
    "booty", "CuteLittleButts", "HotMoms", "cumshots", "Ebony", "TinyTits", "redheads", "smallboobs", "boobbounce", "girlsinyogapants", 
    "IWantToSuckCock", "homemadexxx", "bdsm", "CollegeAmateurs", "Orgasms", "DaughterTraining", "BlowjobGirls", "obsf", "SlimThick", 
    "FaceFuck", "Miakhalifa", "tipofmypenis", "girlswhoride", "AsianNSFW", "NSFWverifiedamateurs", "maturemilf", "SpreadEm", 
    "CamSluts", "BrownHotties", "hugeboobs", "iwanttobeher", "fitgirls", "BornToBeFucked", "Doggystyle_NSFW", "AdorableNudes", 
    "JustFriendsHavingFun", "DegradingHoles", "bodyperfection", "amihot", "rearpussy", "couplesgonewild", "gwpublic", "wifesharing", 
    "futanari", "IndiansGoneWild", "MassiveTitsnAss", "BubbleButts", "AmateurPorn", "porn_gifs", "tits", "snapleaks", "transporn", 
    "GirlswithGlasses", "MassiveCock", "holdmyfeedingtube", "MasturbationGoneWild", "cheatingwives", "assholegonewild", "petite", 
    "PerfectBody", "amateur_milfs", "Step_Fantasy_GIFs", "Nipples", "ebonyhomemade", "WetPussys", "GWCouples", "Hotchickswithtattoos", 
    "BDSMGW", "short_porn", "confessionsgonewild", "confesiones_intimas", "DirtyConfession", "Incestconfessions", "SextStories"
]))

# List for subreddits with a significant mix of SFW and NSFW content
MIXED_CONTENT_SUBREDDITS = list(set([
    "AskReddit", "worldnews", "todayilearned", "gaming", "AmItheAsshole", "tifu", "wallstreetbets", "nosleep", "relationship_advice", 
    "creepy", "interestingasfuck", "Tinder", "unpopularopinion", "PublicFreakout", "offmychest", "confession", "politics", "WTF", 
    "fights", "MakeMeSuffer", "MorbidReality", "iamatotalpieceofshit", "trashy", "NoahGetTheBoat", "Dhaka", "bangladesh", "LetsNotMeet", 
    "TrueCrime", "exmuslim", "Drugs", "AskRedditAfterDark", "worldpolitics", "TikTokCringe"
]))


try:
    # --- Ingest Data ---
    print("\n--- Step 1: Ingesting from clearly SFW subreddits ---")
    normal_posts = get_classified_posts(SFW_SUBREDDITS, POSTS_PER_SFW_SUB, "SFW")

    print("\n--- Step 2: Ingesting from clearly NSFW subreddits ---")
    anomaly_posts = get_classified_posts(NSFW_SUBREDDITS, POSTS_PER_NSFW_SUB, "NSFW")

    print("\n--- Step 3: Scanning mixed-content subreddits ---")
    mixed_sfw, mixed_nsfw = get_mixed_posts(MIXED_CONTENT_SUBREDDITS, POSTS_TO_SCAN_PER_MIXED_SUB)
    
    # Add the posts found from the mixed scan to our main lists
    normal_posts.extend(mixed_sfw)
    anomaly_posts.extend(mixed_nsfw)

    # --- Combine and Save ---
    total_normal = len(normal_posts)
    total_anomaly = len(anomaly_posts)
    
    if total_anomaly == 0:
        print("\n🚨 CRITICAL WARNING: No NSFW posts were found. The model cannot be trained correctly.")

    all_posts_data = normal_posts + anomaly_posts
    df = pd.DataFrame(all_posts_data, columns=[
        "title", "body", "score", "num_comments", "subreddit", 
        "classification",  # Original binary classification
        "safety", "toxicity", "sentiment", "topic", "engagement"  # Multi-label classifications
    ])
    
    df.dropna(subset=['title', 'body'], inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # --- Overwrite old data for a fresh run ---
    data_dir = "data"
    output_file = os.path.join(data_dir, "raw_posts.csv")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Explicitly delete the old CSV file if it exists
    if os.path.exists(output_file):
        print(f"\nFound old data file. Deleting {output_file} to create a fresh dataset.")
        os.remove(output_file)
    
    # Save the new dataframe to a fresh CSV file
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print("✅ Multi-Label Data Ingestion Complete!")
    print(f"Total SFW (Normal) Posts Fetched: {total_normal}")
    print(f"Total NSFW (Anomaly) Posts Fetched: {total_anomaly}")
    print(f"Total Usable Posts Saved to CSV: {len(df)}")
    
    # Print multi-label statistics
    print("\n--- Multi-Label Classification Statistics ---")
    for category in ['safety', 'toxicity', 'sentiment', 'topic', 'engagement']:
        print(f"{category.title()} distribution:")
        print(df[category].value_counts().to_dict())
        print()
    
    print("="*50)

except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
