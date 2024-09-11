import fitz  # PyMuPDF
import asyncio
from fastapi import HTTPException
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from fastapi.middleware.cors import CORSMiddleware


# Ensure necessary NLTK resources are downloaded
def download_nltk_resources():
    # nltk.download()
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# download_nltk_resources()

def preprocess_cv_text(cv_text):
    # Tokenize the text
    tokens = word_tokenize(cv_text)

    # Convert to lower case
    tokens = [token.lower() for token in tokens]

    # Remove punctuation and non-alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Custom redundant words specific to CVs (can be modified)
    custom_redundant_words = {
    'curriculum', 'vitae', 'resume', 'application', 'applicant',
    'employment', 'professional', 'career', 'job', 'experience',
    'work', 'history', 'qualification', 'academic', 'education',
    'school', 'college', 'university', 'degree', 'diploma',
    'certification', 'certified', 'training', 'course', 'courses',
    'skills', 'abilities', 'projects', 'objective', 'summary',
    'profile', 'personal', 'details', 'address', 'contact',
    'phone', 'email', 'e-mail', 'name', 'birthdate', 'gender',
    'nationality', 'status', 'marital', 'language', 'languages',
    'proficient', 'fluent', 'native', 'speaker', 'hobbies',
    'interests', 'references', 'available', 'request', 'portfolio',
    'linkedin', 'github', 'social', 'media', 'website', 'blog',
    'online', 'page', 'http', 'https', 'www', 'Achieved','Administered',
    'Advanced','Advised','Advocated','Analyzed','Arranged','Assembled',
    'Assessed','Assigned','Assisted','Attained','Authored','Balanced',
    'Boosted','Built','Calculated','Centralized','Chaired','Changed',
    'Collaborated','Communicated','Compared','Compiled','Completed',
    'Composed','Conducted','Consolidated','Constructed','Consulted',
    'Controlled','Converted','Coordinated','Created','Cultivated',
    'Customized','Decreased','Defined','Delivered','Demonstrated',
    'Designed','Developed','Devised','Directed','Discovered','Doubled',
    'Drafted','Drove','Edited','Educated','Enhanced','Established',
    'Estimated','Evaluated','Examined','Executed','Expanded',
    'Expedited','Experimented','Explained','Explored','Extended',
    'Facilitated','Financed','Fixed','Focused','Forecasted','Formed',
    'Formulated','Fostered','Founded','Gained','Generated','Guided',
    'Handled','Headed','Identified','Illustrated','Implemented',
    'Improved','Improvised','Increased','Influenced','Informed',
    'Initiated','Innovated','Inspected','Installed','Instituted',
    'Integrated','Interpreted','Introduced','Invented','Investigated',
    'Launched','Led','Leveraged','Maintained','Managed','Mapped',
    'Marketed','Measured','Mediated','Modeled','Modified','Monitored',
    'Motivated','Negotiated','Operated','Orchestrated','Organized',
    'Outlined','Overhauled','Oversaw','Participated','Performed',
    'Persuaded','Planned','Prepared','Presented','Processed',
    'Produced','Programmed','Promoted','Proposed','Proved','Provided',
    'Publicized','Published','Purchased','Recommended','Reconciled',
    'Recruited','Redesigned','Reduced','Refined','Regulated',
    'Reinforced','Reorganized','Replaced','Reported','Represented',
    'Researched','Resolved','Responded','Restored','Restructured',
    'Revamped','Reviewed','Revitalized','Saved','Scheduled','Secured',
    'Selected','Simplified','Solved','Spearheaded','Specialized',
    'Standardized','Started','Streamlined','Strengthened','Structured',
    'Studied','Submitted','Succeeded','Summarized','Supervised',
    'Supported','Surpassed','Surveyed','Sustained','Tailored',
    'Targeted','Taught','Tested','Tracked','Trained','Transformed',
    'Translated','Troubleshot','Tuned','Uncovered','Undertook',
    'Unified','Updated','Upgraded','Utilized','Validated','Verified',
    'Visualized','Widened'
}

    final_tokens = [token for token in lemmatized_tokens if token not in custom_redundant_words]

    return final_tokens


async def extract_keywords_from_cv(file_path: str):
    try:
        # Open the PDF file from the path
        await asyncio.sleep(0)  # Yield control to keep things async-friendly
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()

        # # Dummy example of keyword extraction logic
        # keywords = set(word for word in text.split() if len(word) > 2)  # simplistic filter for keywords
        keywords = preprocess_cv_text(text)
        # categories = categorize_skills(keywords)
        
        return list(keywords)
        # return categories
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))