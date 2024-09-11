from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer

data = {
    "JavaScript, TypeScript, C#, Ruby, Go, Rust, Kotlin, Swift, PHP, Perl, R, Scala, Objective-C, Dart, Haskell, Lua, MATLAB, Shell scripting, Groovy, Assembly, F#, Erlang, Clojure, Visual Basic, Ada, Fortran, Prolog, COBOL,Python, Java, C++": "Programming",
    "Budgeting, Forecasting, P&L Management": "Finance",
    "Photoshop, Illustrator, InDesign": "Design",
    "Risk management, Decision making, Strategic thinking, Time management, Resource allocation, Change management, Conflict resolution, Performance monitoring, Budget management, Quality control, Operations management, Staff development, Problem solving, Communication, Adaptability, Innovation, Negotiation, Stakeholder management, Process improvement, Coaching, Mentoring, Crisis management, Organizational skills, Data analysis, Customer relationship management,Project planning, Team leadership": "Management",
    "Machine Learning, Data Mining, Neural Networks, Deep Learning, Natural Language Processing,NLP, Computer Vision, Big Data, Statistical Analysis, Predictive Analytics, Data Visualization, Data Warehousing, Business Intelligence, Decision Trees, Random Forests, Gradient Boosting Machines, Support Vector Machines,SVM, Clustering, Dimensionality Reduction, Principal Component Analysis,PCA, t-Distributed Stochastic Neighbor Embedding,t-SNE, K-means Clustering, Hierarchical Clustering, DBSCAN, Association Rules, Time Series Analysis, Anomaly Detection, Recommender Systems, Text Mining, Sentiment Analysis, Topic Modeling, Bayesian Statistics, Monte Carlo Simulations, Markov Chains, Artificial Neural Networks,ANN, Convolutional Neural Networks ,CNN, Recurrent Neural Networks ,RNN, Long Short-Term Memory Networks ,LSTM, Generative Adversarial Networks, GAN, Reinforcement Learning, Feature Engineering, Feature Selection, Model Validation, Cross-Validation, Hyperparameter Tuning, Ensemble Methods, Bagging, Boosting, Model Deployment, Big Data,Hadoop,Spark, Data Ethics, Data Governance, Data Security, Data Privacy, Automated Machine Learning (AutoML), Explainable AI (XAI), AI Fairness, Bias Detection, Blockchain in Data Science": "Data Science",
    "Active listening, Verbal communication, Written communication, Interpersonal communication, Persuasion, Conflict resolution, Empathy, Non-verbal communication, Report writing, Storytelling, Feedback delivery, Assertiveness, Customer service, Team communication, Media relations, Editing, Clarity and conciseness, Diplomacy, Influence, Cross-cultural communication, Networking, Crisis communication, Technical writing,Public speaking, Presentation, Negotiation": "Communication",
    "Electrical circuits, System design, PLC programming": "Engineering",
    "Digital marketing, Email marketing, Market research, Brand management, Product marketing, Data analysis, Advertising, Public relations, Event planning, Customer segmentation, Lead generation, Conversion optimization, Strategic planning, Marketing automation, PPC (Pay Per Click), Graphic design, Video production, Influencer marketing, CRM (Customer Relationship Management), Copywriting, Analytics and reporting, E-commerce marketing, User experience (UX) design,Social media, Content creation, SEO": "Marketing",
    "Litigation, Intellectual property, Legal research, Legal writing, Mediation, Arbitration, Employment law, Environmental law, Family law, Criminal law, Real estate law, Tax law, Bankruptcy law, Civil rights law, International law, Mergers and acquisitions, Personal injury law, Estate planning, Legal due diligence, Public speaking, Trial preparation, Legal advice, Data privacy,Corporate law, Contract negotiation, Compliance": "Legal",
    "Performance management, Conflict resolution, Compensation and benefits, Talent management, Succession planning, Labor relations, Organizational development, Workforce planning, Diversity and inclusion, HR compliance, Change management, Employee relations, HR strategy, Onboarding, HR analytics, Employee retention, Leadership development, HR policy formulation,Recruitment, Training, Employee engagement": "Human Resources",
    "Nancy, ServiceStack,AdonisJs, Total.js,Akka HTTP,Phalcon, Zend Framework,Vert.x, Dropwizard,Revel, Echo,Rocket, Warp,Bottle, CherryPy, Falcon, Hug,Node.js, Django, Flask, .NET,Express.js, NestJS, Koa.js, Meteor.js, Sails.js, Django, Flask, FastAPI, Tornado, Ruby on Rails, Sinatra, Spring Boot, Hibernate, Apache Struts,Laravel, Symfony, CodeIgniter,ASP.NET Core, Entity Framework,Go,Gin, Beego,Elixir,Phoenix,Scala,Play Framework,Rust,Actix Web": "Backend-Development",
    "React, Angular, Vue.js, Svelte, Ember.js, Backbone.js, Polymer, Aurelia, HTML, CSS,Angular, Stencil.js,Bootstrap, Tailwind CSS, Foundation, Bulma, Materialize,Lodash, Underscore.js, Redux, MobX, Vuex, NgRx": "Frontend-Development",
    "JUnit, Selenium, Cypress, Postman, Jest, Mocha, Jasmine,TestNG, NUnit, QUnit, Ava, TestCafe, Robot Framework, SpecFlow, xUnit.net, Karma, Protractor, SoapUI, Appium, LoadRunner": "Testing",
    "Maven, Gradle, Ant, Make, CMake, Bazel, MSBuild, SBT, Gulp, Webpack, Rollup, Parcel":"Build Tools"
}

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def train_and_evaluate(data, classifier='svm'):
    X = list(data.keys())
    y = list(data.values())

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.25, random_state=42)

    if classifier == 'svm':
        model = SVC(kernel='linear', probability=True)
    elif classifier == 'logistic':
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError("Unsupported classifier type")

    model.fit(X_train, y_train)
    return model, vectorizer

def predict_category(model, vectorizer, keywords):
    keywords_vectorized = vectorizer.transform([keywords])
    category = model.predict(keywords_vectorized)
    return category[0]

def process_keywords(args):
    model, vectorizer, keywords = args
    return predict_category(model, vectorizer, keywords)

def categorize_skills(keywords_list, classifier_type):
    model, vectorizer = train_and_evaluate(data, classifier=classifier_type)
    counter_map = {}

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(process_keywords, [(model, vectorizer, k) for k in keywords_list]))

    for category in results:
        if category in counter_map:
            counter_map[category] += 1
        else:
            counter_map[category] = 1

    return counter_map
