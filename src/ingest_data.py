import praw
import pandas as pd
import os
from dotenv import load_dotenv
import time

load_dotenv()

# --- Configuration ---
# You can adjust these numbers. A larger number means a better dataset, but it will take longer to run.
POSTS_PER_SFW_SUB = 100
POSTS_PER_NSFW_SUB = 150 
# How many posts to scan from mixed subreddits to find SFW/NSFW content
POSTS_TO_SCAN_PER_MIXED_SUB = 200

# --- Get Credentials ---
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

if not all([client_id, client_secret, user_agent]):
    raise ValueError("Reddit credentials not set in .env file.")

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

def get_classified_posts(subreddit_list, posts_per_sub, classification_label):
    """Fetches posts from subreddits that are assumed to be 100% SFW or 100% NSFW."""
    all_posts = []
    for sub_name in subreddit_list:
        try:
            print(f"Fetching {posts_per_sub} posts from r/{sub_name}...")
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=posts_per_sub):
                all_posts.append([post.title, post.selftext, post.score, post.num_comments, sub_name, classification_label])
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ Could not fetch from r/{sub_name}. Error: {e}")
    return all_posts

def get_mixed_posts(subreddit_list, posts_to_scan):
    """
    Scans posts from mixed-content subreddits and classifies each post individually
    based on its 'over_18' (NSFW) flag.
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

# --- Define Your Subreddit Lists ---
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
    "PeopleFuckingDying", "fantasyfootball", "yoga", "cursedcomments", "GifRecipes", "MurderedByWords", "harrypotter", "zelda", "Marvel", "raspberry_pi", "Perfectfit"
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
    df = pd.DataFrame(all_posts_data, columns=["title", "body", "score", "num_comments", "subreddit", "classification"])
    
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
    
    print("\n" + "="*30)
    print("✅ Data Ingestion Complete!")
    print(f"Total SFW (Normal) Posts Fetched: {total_normal}")
    print(f"Total NSFW (Anomaly) Posts Fetched: {total_anomaly}")
    print(f"Total Usable Posts Saved to CSV: {len(df)}")
    print("="*30)

except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")