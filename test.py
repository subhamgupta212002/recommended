import streamlit as st
import joblib
import pandas as pd

# Load models
spam_model = joblib.load('spam_classifier.pkl')
language_model = joblib.load('lang_det.pkl')
news_model = joblib.load('news_cat.pkl')
review_model = joblib.load('review.pkl')

st.set_page_config(layout='wide')

# Page title
st.markdown("""
    <h1 style='text-align: center; color: #9B59B6;'>
        ✨💡 LENS eXpert 💡✨ <br>
        <span style='font-size: 24px; color: #566573;'>(NLP Suits)</span>
    </h1>
    """, unsafe_allow_html=True)

# Tabs with tracking
selected_tab = st.selectbox('Select a Module:', ['✉️ Spam Classifier', '🗣️ Language Detection', '🍔 Food Review Sentiment', '🗞️ News Classification', '🧑‍🔬 Profile','🔍 Movie Recommendation System'])

# Sidebar - Always Visible Content🧑
st.sidebar.image('IMG_3441-01-01.jpeg')

with st.sidebar.expander('ℹ️ About us'):
    st.write('We are a group of students trying to understand the concept of NLP')
    st.write("""Welcome to 🔍 LENS eXpert (NLP Suits) — **your all-in-one Natural Language Processing** toolkit designed to make text analysis easy, fast, and smart!

Our platform brings together four powerful NLP solutions under one roof to help you efficiently process and understand language data in the real world.

✨ Our Modules:
- ✉️ **Spam Classifier** :
We protect your inbox and communication channels by accurately detecting and filtering out unwanted spam messages. Whether it's emails, SMS, or social media comments, our spam classifier keeps your space clean and safe.

- **🗣️ Language Detection** :
With just a few words, our system quickly detects the language of your text, enabling seamless processing of multilingual content. Perfect for global businesses, chatbots, and multilingual platforms.

- **🍔 Food Review Sentiment** :
Discover what people really think about your food! Our sentiment analysis tool scans customer reviews and determines whether they are positive, negative, or neutral, helping restaurants and businesses improve their services and offerings.

- **🗞️ News Classification** :
Stay organized and informed with our news classification module. It automatically categorizes news articles by topics, helping you sort and process vast amounts of information quickly and accurately.''')
             """)

with st.sidebar.expander('🌐 Contact'):
    st.write('📞 9101121227')
    st.write('📧 guptasubham@797gmail.com')

with st.sidebar.expander('Help'):
    st.write('subham')


# Module Pages
if selected_tab == '✉️ Spam Classifier':
    st.header('✉️ Spam Classifier')
    st.write('A Spam Classifier is a system that automatically identifies whether a message is spam (unwanted or harmful content) or ham (legitimate content). It helps in filtering out irrelevant or potentially dangerous messages like ads, phishing links, or fraud attempts.')
    msg1 = st.text_input('Enter Msg', key='msg1')
    if st.button('Prediction', key='b1'):
        pre = spam_model.predict([msg1])
        if pre[0] == 0:
            st.image('spam.jpg')
        else:
            st.image('not_spam.png')

    uploader_file1 = st.file_uploader('Upload file containing bulk messages', type=['csv', 'txt'], key='uploader_file1')

    if uploader_file1:
        df_spam = pd.read_csv(uploader_file1, header=None, names=['Msg'])
        pred = spam_model.predict(df_spam.Msg)
        df_spam.index = range(1, df_spam.shape[0] + 1)
        df_spam['Prediction'] = pred
        df_spam['Prediction'] = df_spam['Prediction'].map({0: 'Spam', 1: 'Not Spam'})
        st.dataframe(df_spam)

elif selected_tab == '🗣️ Language Detection':
    st.header('🗣️ Language Detection')
    st.write('Language Detection is the process of automatically identifying the language in which a given piece of text is written. This is a crucial first step in many Natural Language Processing (NLP) pipelines, especially when working with multilingual data.')
    msg2 = st.text_input('Enter Msg', key='msg2')
    if st.button('Detect', key='b2'):
        pre = language_model.predict([msg2])
        st.success(pre)

    uploader_file2 = st.file_uploader('Upload file containing bulk messages', type=['csv', 'txt'], key='uploader_file2')

    if uploader_file2:
        df_lang = pd.read_csv(uploader_file2, header=None, names=['Msg'])
        pred = language_model.predict(df_lang.Msg)
        df_lang.index = range(1, df_lang.shape[0] + 1)
        df_lang['Prediction'] = pred
        st.dataframe(df_lang)

elif selected_tab == '🍔 Food Review Sentiment':
    st.header('🍔 Food Review Sentiment')
    st.write('Food Review Sentiment Analysis automatically determines whether a food review expresses a positive, negative, or neutral opinion. It helps businesses and users quickly understand public sentiment toward food, restaurants, or services.')
    msg3 = st.text_input('Enter Msg', key='msg3')
    if st.button('Review', key='b3'):
        pre = review_model.predict([msg3])
        if pre[0] == 0:
            st.warning('😞 Not Satisfied')
        else:
            st.success('😃 Satisfied')

    uploader_file3 = st.file_uploader('Upload file containing bulk messages', type=['csv', 'txt'], key='uploader_file3')

    if uploader_file3:
        df_review = pd.read_csv(uploader_file3, header=None, names=['Msg'])
        pred = review_model.predict(df_review.Msg)
        df_review.index = range(1, df_review.shape[0] + 1)
        df_review['Prediction'] = pred
        df_review['Prediction'] = df_review['Prediction'].map({0: '😞 Not Satisfied', 1: '😃 Satisfied'})
        st.dataframe(df_review)

elif selected_tab == '🗞️ News Classification':
    st.image('download.jpeg')

elif selected_tab == "🧑‍🔬 Profile":
    st.title("Subham Gupta")
    st.write("---")

    st.header("👤 About Me")
    st.write("""**Hello**, my name is Subham Gupta. I’m from Assam and I’ve completed my Bachelor's 
                 degree in Business Administration from Greater Noida. Currently,
                  I’m upskilling in the field of Data Science, and I have recently completed a Data Analyst course. 
                 During this course, I gained hands-on experience with tools such as SQL, Python, and Power BI,
                  along with knowledge of data visualization and statistical analysis. 
                 I am passionate about using data to solve real-world problems and eager to start my career as a data analyst.""")

    st.header("🧰 Skills")
    st.write("""
        - **Data Analysis**: Proficient in SQL Power BI, and Python for data analysis and reporting.
        - **Data Visualizations**: Experienced in creating dashboards and visual reports using Matplotlib, Seaborn, Power BI
          and Tableau.
        - **Database Management**: Skilled in SOL, database design, and management.
        - **Programming**: Knowledge of Python for data manipulation and analysis, with experience in
          libraries such as Pandas, Numpy and Statistics.
        - **Data Cleaning**: Well-versed in techniques to prepare raw data for analysis, including:
        - Handling Missing Values
        - Correcting Data Errors Managing
        - Data Duplicacy Performing Data
        - Transformations
             """)

    
    st.header("🛠️ Projects")
    st.subheader('Python Project Using Database - ')
    st.write("""- **BANKING AUTOMATION MANAGEMENT -** A project designed for the banking sector, featuring user and admin interfaces. It
facilitates essential banking operations such as account creation, deposits, fund transfers, and withdrawals. Users can also
update personal details like name, phone number, Aadhaar, and email. All data is securely managed and stored in a backend
database.""")
    st.markdown("[🔗 **BANKING AUTOMATION MANAGEMENT -**](https://github.com/subhamgupta212002/my-bank)")
    
    st.write("""- **RESTAURANT BILLING MANAGEMENT -**- A billing application used in retail stores and restaurants to calculate total bills. It
allows for the creation of product lists with associated prices and automates the billing process for improved e iciency and
accuracy.""")
    st.markdown("[🔗 **RESTAURANT BILLING MANAGEMENT -**](https://github.com/subhamgupta212002/restaurant)")

    st.subheader('Power BI Project- ')
    st.write("""- **IPL_ANALYSIS DASHBOARD -**- A Power BI project that provides interactive insights into IPL matches and player performances
across different years. Key metrics include wins, losses, trophy winners, most sixes, most fours, most centuries, highest wickettakers, and top run-scorers.""")
    st.markdown("- [🔗 **IPL_ANALYSIS DASHBOARD -**](https://app.powerbi.com/groups/me/reports/15fa0400-44f1-4bfb-920d-67c3068b4bfc/83ee8f6c04805007709d?experience=power-bi)")

    st.write("""- **HR ANALYSIS DASHBOARD -**- A Power BI dashboard commonly used in HR analytics to gain insights into employee data.includes
visualizations on employee count, salary distribution, age groups, departments, overtime, education levels, and job
satisfaction.""")
    st.markdown("- [🔗 **HR ANALYSIS DASHBOARD -**](https://app.powerbi.com/groups/me/reports/2801758f-391f-4d5d-be52-ce0ef2127939/4f3e9f78923f1b29def3?experience=power-bi)")

    
    st.write("""- **GLOBAL WATER CONSUMPTION -**- A Power BI dashboard that highlights water consumption trends across the top 5 countries by
year. It analyzes usage in sectors such as agriculture, household, and industry, and includes metrics like per capita
consumption, rainfall impact, and groundwater depletion rates.""")
    st.markdown("- [🔗 **GLOBAL WATER CONSUMPTION -**](https://app.powerbi.com/groups/me/reports/d2444240-699b-40ac-8ccb-3fa4f90fa78b/f32a39d947bcd191db47?experience=power-bi)")

    
    st.write("""- **Sachin Tendulkar ODI Power BI Dashboard -**- This Power BI dashboard offers a dynamic and visually engaging summary of Sachin Tendulkar’s ODI career. 
             It showcases key highlights like total matches (463), total runs (15,203), centuries (34), fifties (116), wickets (154), and toss stats. The dashboard features a 
             venue-wise performance chart, pinpointing top-scoring grounds such as Dhaka, Sharjah, and Colombo. With easy navigation to year-wise and century-specific insights, this dashboard delivers a complete,
             interactive view of Sachin’s ODI legacy in a clean, intuitive layout.""")
    st.markdown("- [🔗 **Sachin Tendulkar ODI Power BI Dashboard -**](https://app.powerbi.com/groups/me/reports/157d04c6-867c-4801-8676-e8d579f20dcf/501fac6c46f46559b1b5?experience=power-bi)")

    st.write("""- **Blinkit Sales Dashboard -**- This Power BI dashboard provides an interactive overview of Blinkit's sales performance. It highlights key metrics such as Total Sales ($1.20M), Average Sales ($141), 
             Total Items (8523), and Average Rating (3.9). The dashboard features a filter panel to refine data by Outlet Location, Size, and Item Type. Visuals include item category breakdowns, fat content analysis, 
             and outlet-wise fat distribution. Clean, user-friendly, and focused on fast decision-making.
""")
    st.image('pbi.jpg')
    st.markdown("- [🔗 **Blinkit Sales Dashboard -**](https://app.powerbi.com/groups/me/reports/4f0bfc3d-560f-40be-b565-3d753f3da3ef/13c14fd88c047a5d0e20?experience=power-bi)")
    
    st.write("""- **Olympic Games (1896–2016) Dashboard -**-This Power BI dashboard provides a visual summary of Olympic medal counts from 1896 to 2016. It highlights medal distribution by country, with Germany and Greece leading.
              The dashboard also lists top medal-winning athletes, such as Hermann Otto Ludwig Weingärtner with 6 medals. Additional filters and charts offer insights into medal counts by gender, total medals, and sports categories.
              The layout is clean, interactive, and easy to explore year-wise trends.""")
    st.markdown("- [🔗 **Olympic Games (1896–2016) Dashboard -**](https://app.powerbi.com/groups/me/reports/eae5e3f0-e022-44a1-8323-3caebe229c43/ea8311a4eb77d8f44461?experience=power-bi)")


    st.subheader('**Meachine Learning -** ')
    st.write("""
            - **💡 LENS_R eXpert - NLP Suite-**
            Developed a multi-functional NLP web application using Streamlit that integrates four key text classification tools: Spam Detection, Language Detection,
            Food Review Sentiment Analysis, and News Classification. Enabled both single and bulk text processing with interactive dashboards.
            - **Key Features**:
            - Built ML models for spam classification, sentiment analysis, and language detection.
            - Designed a user-friendly interface with modular navigation and profile management.
            - Incorporated bulk file uploads for large-scale text predictions.
            - Provided dynamic visual feedback and personalized profile sections.
             """)
    st.markdown("- [🔗 **💡 LENS_R eXpert -**](https://github.com/subhamgupta212002/recommended/blob/main/test.py)")
    st.markdown("- [ **Streamlit -**](https://recommend2025.streamlit.app/)")
    
    st.header("🎓 Education")
    st.write("""**- Bachelor of Business Administrative (BBA)**- Llyod Group of Institute Management and Technology, Greater Noida, Graduated
2025| 2024 - Level up My skill: Data science -Data Analysis, Data Management. Python Programming, Statistics.""")
    
elif selected_tab == "🔍 Movie Recommendation System":
    import streamlit as st
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    import requests

    # Load and prepare the data
    @st.cache_data
    def load_data():
        df = pd.read_csv('movies_content.csv')
        df2 = df[['movie_id', 'description', 'language', 'director', 'cast', 'genre', 'name']]
        df3 = df2.dropna()
        df4 = df3.reset_index(drop=True)
        df5 = df4[['movie_id', 'name']]
        df5['tag'] = df4['description'] + " " + df4['language'] + " " + df4['director'] + " " + df4['cast'] + " " + df4['genre']
        return df5

    df5 = load_data()

    # TF-IDF Vectorization
    tv = TfidfVectorizer(lowercase=True)
    vectors = tv.fit_transform(df5.tag).toarray()

    # Nearest Neighbors Model
    model = NearestNeighbors(metric='cosine')
    model.fit(vectors, vectors)

    # Fetch movie poster from OMDb API
    def fetch_movie_poster(imdb_id, api_key='8fecb11c'):
        api_url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
        try:
            resp = requests.get(api_url)
            if resp.status_code == 200:
                data = resp.json()
                if 'Poster' in data and data['Poster'] != 'N/A':
                    return data['Poster']
                else:
                    return "https://via.placeholder.com/150?text=No+Poster"
            else:
                return "https://via.placeholder.com/150?text=Error"
        except Exception:
            return "https://via.placeholder.com/150?text=Error"

    # Recommend movies function
    def recommend_movies(movie_name, n_recommendations=5):
        if movie_name not in df5['name'].values:
            st.error(f"Movie '{movie_name}' not found in the dataset.")
            return []

        index = df5[df5['name'] == movie_name].index[0]
        distances, indexes = model.kneighbors([vectors[index]], n_neighbors=n_recommendations + 1)

        recommendations = []
        for i in indexes[0][1:]:
            movie_id = df5.loc[i, 'movie_id']
            movie_name_rec = df5.loc[i, 'name']
            poster_url = fetch_movie_poster(movie_id)

            recommendations.append({'movie_id': movie_id, 'name': movie_name_rec, 'poster': poster_url})
        return recommendations

    # Streamlit UI
    st.title("🎬 Movie Recommendation System")

    selected_movie = st.selectbox("Select a movie to get recommendations:", df5['name'].values)

    if st.button("Recommend"):
        with st.spinner('Fetching recommendations...'):
         results = recommend_movies(selected_movie)

        if results:
            st.subheader(f"Recommended movies for '{selected_movie}':")
            cols = st.columns(len(results))
            for idx, movie in enumerate(results):
                with cols[idx]:
                    st.image(movie['poster'], width=150)
                    st.caption(f"{movie['name']}")

    st.header("📞 Contact")
    st.write("""📧 guptasubham797@gmail.com | 📱 +91-9101121227""")

    st.write("---")
    st.write("Thank You")
