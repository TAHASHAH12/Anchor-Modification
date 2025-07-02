import streamlit as st
import pandas as pd
import requests
from openai import OpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import time
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

st.set_page_config(
    page_title="SEO Link Opportunity Analyzer",
    page_icon="🔗",
    layout="wide"
)

if 'opportunity_data' not in st.session_state:
    st.session_state.opportunity_data = None
if 'internal_data' not in st.session_state:
    st.session_state.internal_data = None

class ContentFetcher:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def fetch_page_content(self, url: str) -> Dict:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            raw_text = soup.get_text()
            
            lines = (line.strip() for line in raw_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)
            
            title = soup.find('title')
            title_text = title.get_text() if title else ""
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            desc_text = meta_desc.get('content') if meta_desc else ""
            
            content_summary = self.extract_key_content(clean_text[:4000], title_text, desc_text)
            
            return {
                'title': title_text,
                'meta_description': desc_text,
                'content': clean_text[:4000],
                'summary': content_summary,
                'word_count': len(clean_text.split()),
                'success': True
            }
            
        except Exception as e:
            return {
                'title': '',
                'meta_description': '',
                'content': '',
                'summary': f'Error fetching content: {str(e)}',
                'word_count': 0,
                'success': False
            }
    
    def extract_key_content(self, content: str, title: str, meta_desc: str) -> str:
        prompt = f"""
        Analyze the following webpage content and provide a concise summary of the main topics, themes, and key information:
        
        Title: {title}
        Meta Description: {meta_desc}
        Content: {content}
        
        Please provide:
        1. Main topic/theme
        2. Key points covered
        3. Target audience/purpose
        4. Important keywords and phrases
        
        Keep the summary under 300 words.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def analyze_topic_relevance(self, page_content: Dict, target_topic: str) -> Dict:
        """Analyze how well the page content matches the target topic"""
        prompt = f"""
        Analyze how relevant this webpage content is to the target topic: "{target_topic}"
        
        Page Information:
        Title: {page_content.get('title', '')}
        Meta Description: {page_content.get('meta_description', '')}
        Content Summary: {page_content.get('summary', '')}
        
        Target Topic: {target_topic}
        
        Please provide:
        1. Relevance Score (0-10): How relevant is this page to the target topic?
        2. Topic Match Analysis: Explain the connection between page content and target topic
        3. Key Matching Elements: What specific elements on the page relate to the target topic?
        4. Content Context: What is the overall context/theme of this page?
        
        Format your response as:
        Relevance Score: [0-10]
        Topic Match: [brief analysis]
        Matching Elements: [key elements]
        Context: [page context]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Parse the response to extract relevance score
            relevance_score = 0
            lines = analysis.split('\n')
            for line in lines:
                if 'Relevance Score:' in line:
                    try:
                        score_text = line.split(':')[1].strip()
                        relevance_score = float(re.findall(r'\d+(?:\.\d+)?', score_text)[0])
                        break
                    except:
                        pass
            
            return {
                'relevance_score': relevance_score,
                'analysis': analysis,
                'success': True
            }
            
        except Exception as e:
            return {
                'relevance_score': 0,
                'analysis': f"Topic analysis failed: {str(e)}",
                'success': False
            }
    
    def suggest_topic_based_anchor_text(self, page_content: Dict, target_url: str, target_topic: str, topic_analysis: Dict) -> List[str]:
        """Generate anchor text suggestions based on target topic and page relevance"""
        relevance_score = topic_analysis.get('relevance_score', 0)
        topic_analysis_text = topic_analysis.get('analysis', '')
        
        prompt = f"""
        Create EXACTLY 2 anchor text suggestions for a link to this URL, specifically focused on the target topic: "{target_topic}"
        
        Page Information:
        URL: {target_url}
        Title: {page_content.get('title', '')}
        Meta Description: {page_content.get('meta_description', '')}
        Content Summary: {page_content.get('summary', '')}
        
        Target Topic: {target_topic}
        Topic Relevance Score: {relevance_score}/10
        Topic Analysis: {topic_analysis_text}
        
        Create anchor text that:
        1. Relates directly to the target topic "{target_topic}"
        2. Is natural and contextually relevant to the page content
        3. Matches the language and tone of the page
        4. Is 2-6 words long
        5. Is SEO-friendly but not over-optimized
        6. Reflects the topic relevance (if low relevance, make it more generic but still topic-related)
        
        Consider the topic relevance score:
        - High relevance (7-10): Use specific, topic-focused anchor text
        - Medium relevance (4-6): Use broader topic-related anchor text
        - Low relevance (0-3): Use generic but topic-adjacent anchor text
        
        Return ONLY 2 anchor text suggestions, one per line, without numbers, bullets, or any other formatting.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            
            suggestions = response.choices[0].message.content.strip().split('\n')
            clean_suggestions = [s.strip('- ').strip() for s in suggestions if s.strip()][:2]
            
            # Fallback suggestions if AI didn't provide enough
            while len(clean_suggestions) < 2:
                if relevance_score >= 7:
                    clean_suggestions.append(f"{target_topic} guide")
                elif relevance_score >= 4:
                    clean_suggestions.append(f"Learn about {target_topic}")
                else:
                    clean_suggestions.append(f"{target_topic} resource")
            
            return clean_suggestions[:2]
            
        except Exception as e:
            st.error(f"Topic-based anchor text generation failed: {str(e)}")
            # Fallback suggestions based on topic
            return [f"{target_topic} information", f"Learn more about {target_topic}"]

def calculate_text_similarity(text1: str, text2: str) -> float:
    try:
        if not text1.strip() or not text2.strip():
            return 0.0
        
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1
        )
        
        texts = [text1, text2]
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception as e:
        return 0.0

def calculate_keyword_match_score(text: str, keyword: str) -> float:
    if not text or not keyword:
        return 0.0
    
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    exact_matches = text_lower.count(keyword_lower)
    
    keyword_words = keyword_lower.split()
    partial_matches = sum(text_lower.count(word) for word in keyword_words)
    
    text_length = len(text.split())
    if text_length == 0:
        return 0.0
    
    match_score = (exact_matches * 3 + partial_matches) / text_length * 100
    return min(match_score, 10.0)

def calculate_url_similarity(url1: str, url2: str) -> float:
    try:
        parsed1 = urlparse(url1)
        parsed2 = urlparse(url2)
        
        domain1 = parsed1.netloc.lower()
        domain2 = parsed2.netloc.lower()
        path1 = parsed1.path.lower()
        path2 = parsed2.path.lower()
        
        url_text1 = f"{domain1} {path1}".replace('/', ' ').replace('-', ' ').replace('_', ' ')
        url_text2 = f"{domain2} {path2}".replace('/', ' ').replace('-', ' ').replace('_', ' ')
        
        return calculate_text_similarity(url_text1, url_text2)
    except:
        return 0.0

def main():
    st.title("🔗 SEO Link Opportunity Analyzer")
    st.markdown("Analyze link opportunities and generate topic-focused anchor texts using AI content analysis")
    
    with st.sidebar:
        st.header("🔧 API Configuration")
        
        openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        
        st.divider()
        
        st.subheader("⚙️ Settings")
        url_limit = st.slider("Maximum URLs to analyze", min_value=1, max_value=500, value=50)
        
        request_timeout = st.slider("Request timeout (seconds)", min_value=5, max_value=30, value=10)
        
        st.divider()
        
        st.subheader("ℹ️ How Topic-Based Anchors Work")
        st.markdown("""
        **Enhanced Anchor Text Generation:**
        
        1. **Topic Analysis**: AI analyzes each page's relevance to your target topic
        2. **Relevance Scoring**: Pages are scored 0-10 for topic relevance
        3. **Smart Suggestions**: Anchor text is tailored based on:
           - High relevance (7-10): Specific, topic-focused anchors
           - Medium relevance (4-6): Broader topic-related anchors  
           - Low relevance (0-3): Generic but topic-adjacent anchors
        
        This ensures your anchor text is contextually appropriate and SEO-effective!
        """)
    
    if not openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        return
    
    content_fetcher = ContentFetcher(openai_api_key)
    
    tab1, tab2 = st.tabs(["🎯 Topic-Based Anchor Suggestions", "🔍 Opportunity Matching"])
    
    with tab1:
        st.header("🎯 Topic-Based Anchor Text Suggestions")
        st.markdown("Upload opportunity URLs and specify your target topic to get AI-powered, topic-focused anchor text suggestions")
        
        # Topic input section
        st.subheader("🏷️ Target Topic Configuration")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            target_topic = st.text_input(
                "Enter your target topic/theme:",
                placeholder="e.g., 'digital marketing', 'sustainable energy', 'fitness training'",
                help="This topic will be used to generate contextually relevant anchor text suggestions"
            )
        
        with col2:
            st.markdown("**Topic Examples:**")
            st.markdown("• Digital Marketing")
            st.markdown("• Web Development") 
            st.markdown("• Health & Wellness")
            st.markdown("• Financial Planning")
        
        if not target_topic:
            st.info("👆 Please enter your target topic above to proceed with anchor text generation")
            return
        
        st.success(f"✅ Target topic set: **{target_topic}**")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with opportunity URLs",
            type=['csv'],
            key="anchor_upload"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} URLs from CSV")
                
                if not df.empty:
                    st.subheader("Column Selection")
                    url_column = st.selectbox("Select URL column:", df.columns.tolist())
                    
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                    
                    if st.button("🚀 Generate Topic-Based Anchor Suggestions", key="analyze_anchors"):
                        urls_to_process = df[url_column].tolist()[:url_limit]
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results = []
                        
                        for i, url in enumerate(urls_to_process):
                            status_text.text(f"Processing URL {i+1}/{len(urls_to_process)}: {url}")
                            
                            # Fetch page content
                            page_content = content_fetcher.fetch_page_content(url)
                            
                            if page_content['success']:
                                # Analyze topic relevance
                                topic_analysis = content_fetcher.analyze_topic_relevance(page_content, target_topic)
                                
                                # Generate topic-based anchor suggestions
                                anchor_suggestions = content_fetcher.suggest_topic_based_anchor_text(
                                    page_content, url, target_topic, topic_analysis
                                )
                                
                                results.append({
                                    'URL': url,
                                    'Title': page_content['title'][:100] + '...' if len(page_content['title']) > 100 else page_content['title'],
                                    'Word_Count': page_content['word_count'],
                                    'Topic_Relevance_Score': f"{topic_analysis.get('relevance_score', 0)}/10",
                                    'Topic_Analysis': topic_analysis.get('analysis', 'N/A')[:200] + '...' if len(topic_analysis.get('analysis', '')) > 200 else topic_analysis.get('analysis', 'N/A'),
                                    'Anchor_Text_1': anchor_suggestions[0],
                                    'Anchor_Text_2': anchor_suggestions[1],
                                    'Target_Topic': target_topic,
                                    'Status': 'Success'
                                })
                            else:
                                results.append({
                                    'URL': url,
                                    'Title': 'N/A',
                                    'Word_Count': 0,
                                    'Topic_Relevance_Score': '0/10',
                                    'Topic_Analysis': 'Content not accessible',
                                    'Anchor_Text_1': 'Content not accessible',
                                    'Anchor_Text_2': 'Content not accessible',
                                    'Target_Topic': target_topic,
                                    'Status': 'Failed'
                                })
                            
                            progress_bar.progress((i + 1) / len(urls_to_process))
                            time.sleep(1)
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            st.success(f"✅ Generated topic-based anchor text suggestions for {len(results)} URLs")
                            
                            success_count = len([r for r in results if r['Status'] == 'Success'])
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total URLs", len(results))
                            with col2:
                                st.metric("Successful", success_count)
                            with col3:
                                st.metric("Failed", len(results) - success_count)
                            with col4:
                                avg_relevance = np.mean([float(r['Topic_Relevance_Score'].split('/')[0]) for r in results if r['Status'] == 'Success'])
                                st.metric("Avg Topic Relevance", f"{avg_relevance:.1f}/10")
                            
                            # Show results with topic relevance insights
                            st.subheader("🎯 Topic-Based Results Analysis")
                            
                            # Categorize by relevance score
                            high_relevance = [r for r in results if r['Status'] == 'Success' and float(r['Topic_Relevance_Score'].split('/')[0]) >= 7]
                            medium_relevance = [r for r in results if r['Status'] == 'Success' and 4 <= float(r['Topic_Relevance_Score'].split('/')[0]) < 7]
                            low_relevance = [r for r in results if r['Status'] == 'Success' and float(r['Topic_Relevance_Score'].split('/')[0]) < 4]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("High Relevance (7-10)", len(high_relevance), help="Pages highly relevant to your topic")
                            with col2:
                                st.metric("Medium Relevance (4-6)", len(medium_relevance), help="Pages moderately relevant to your topic")
                            with col3:
                                st.metric("Low Relevance (0-3)", len(low_relevance), help="Pages with low topic relevance")
                            
                            st.subheader("📊 Detailed Results")
                            st.dataframe(results_df)
                            
                            # Download options
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Complete Results as CSV",
                                data=csv,
                                file_name=f"topic_based_anchor_suggestions_{target_topic.replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                            
                            # High-relevance only download
                            if high_relevance:
                                high_relevance_df = pd.DataFrame(high_relevance)
                                high_csv = high_relevance_df.to_csv(index=False)
                                st.download_button(
                                    label="⭐ Download High-Relevance Results Only",
                                    data=high_csv,
                                    file_name=f"high_relevance_anchors_{target_topic.replace(' ', '_')}.csv",
                                    mime="text/csv"
                                )
                        
                        status_text.text("✅ Topic-based analysis complete!")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.header("🔍 Opportunity Matching")
        st.markdown("Find relevant opportunity URLs based on keyword and URL matching")
        
        input_method = st.radio(
            "Choose input method for internal URLs and keywords:",
            ["Manual Input", "Upload CSV File"],
            key="input_method"
        )
        
        if input_method == "Manual Input":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Upload Opportunity URLs")
                opportunity_file = st.file_uploader(
                    "Upload CSV with opportunity URLs and anchor texts",
                    type=['csv'],
                    key="opportunity_upload"
                )
                
                if opportunity_file is not None:
                    try:
                        opportunity_df = pd.read_csv(opportunity_file)
                        st.success(f"✅ Loaded {len(opportunity_df)} opportunity URLs")
                        
                        if not opportunity_df.empty:
                            opp_url_column = st.selectbox(
                                "Select opportunity URL column:",
                                opportunity_df.columns.tolist(),
                                key="opp_url_col"
                            )
                            anchor_text_column = st.selectbox(
                                "Select anchor text column:",
                                opportunity_df.columns.tolist(),
                                key="anchor_col"
                            )
                            st.session_state.opportunity_data = (opportunity_df, opp_url_column, anchor_text_column)
                            
                    except Exception as e:
                        st.error(f"Error loading opportunity file: {str(e)}")
            
            with col2:
                st.subheader("Internal URL & Keyword")
                internal_url = st.text_input("Enter your internal URL:")
                target_keyword = st.text_input("Enter target keyword:")
                
                if internal_url and target_keyword:
                    st.session_state.internal_data = pd.DataFrame([{
                        'Internal_URL': internal_url,
                        'Target_Keyword': target_keyword
                    }])
        
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Upload Opportunity URLs")
                opportunity_file = st.file_uploader(
                    "Upload CSV with opportunity URLs and anchor texts",
                    type=['csv'],
                    key="opportunity_upload_csv"
                )
                
                if opportunity_file is not None:
                    try:
                        opportunity_df = pd.read_csv(opportunity_file)
                        st.success(f"✅ Loaded {len(opportunity_df)} opportunity URLs")
                        
                        if not opportunity_df.empty:
                            opp_url_column = st.selectbox(
                                "Select opportunity URL column:",
                                opportunity_df.columns.tolist(),
                                key="opp_url_col_csv"
                            )
                            anchor_text_column = st.selectbox(
                                "Select anchor text column:",
                                opportunity_df.columns.tolist(),
                                key="anchor_col_csv"
                            )
                            st.session_state.opportunity_data = (opportunity_df, opp_url_column, anchor_text_column)
                            
                    except Exception as e:
                        st.error(f"Error loading opportunity file: {str(e)}")
            
            with col2:
                st.subheader("Upload Internal URLs & Keywords")
                internal_file = st.file_uploader(
                    "Upload CSV with internal URLs and keywords",
                    type=['csv'],
                    key="internal_upload"
                )
                
                if internal_file is not None:
                    try:
                        internal_df = pd.read_csv(internal_file)
                        st.success(f"✅ Loaded {len(internal_df)} internal URLs")
                        
                        if not internal_df.empty:
                            st.subheader("Column Selection")
                            internal_url_column = st.selectbox(
                                "Select internal URL column:",
                                internal_df.columns.tolist(),
                                key="internal_url_col"
                            )
                            keyword_column = st.selectbox(
                                "Select keyword column:",
                                internal_df.columns.tolist(),
                                key="keyword_col"
                            )
                            
                            st.subheader("Internal URLs Preview")
                            st.dataframe(internal_df.head())
                            
                            processed_internal = internal_df[[internal_url_column, keyword_column]].copy()
                            processed_internal.columns = ['Internal_URL', 'Target_Keyword']
                            st.session_state.internal_data = processed_internal
                            
                    except Exception as e:
                        st.error(f"Error loading internal file: {str(e)}")
        
        if st.button("🔍 Find Relevant Opportunities", key="find_opportunities"):
            opportunity_valid = (st.session_state.opportunity_data is not None and 
                              len(st.session_state.opportunity_data[0]) > 0)
            internal_valid = (st.session_state.internal_data is not None and 
                            not st.session_state.internal_data.empty)
            
            if not opportunity_valid or not internal_valid:
                st.error("Please provide both opportunity URLs and internal URLs with keywords")
            else:
                opportunity_df, opp_url_column, anchor_text_column = st.session_state.opportunity_data
                internal_df = st.session_state.internal_data
                
                all_results = []
                
                for internal_idx, internal_row in internal_df.iterrows():
                    internal_url = internal_row['Internal_URL']
                    target_keyword = internal_row['Target_Keyword']
                    
                    st.info(f"Analyzing internal URL: {internal_url} with keyword: {target_keyword}")
                    
                    opportunity_urls = opportunity_df[opp_url_column].tolist()[:url_limit]
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, opp_idx in enumerate(opportunity_df.index[:url_limit]):
                        opp_url = opportunity_df.loc[opp_idx, opp_url_column]
                        anchor_text = opportunity_df.loc[opp_idx, anchor_text_column]
                        
                        status_text.text(f"Analyzing opportunity {i+1}/{len(opportunity_urls)}: {opp_url}")
                        
                        url_similarity = calculate_url_similarity(internal_url, opp_url)
                        
                        anchor_keyword_match = calculate_keyword_match_score(anchor_text, target_keyword)
                        
                        url_text = f"{opp_url} {anchor_text}"
                        url_keyword_match = calculate_keyword_match_score(url_text, target_keyword)
                        
                        combined_score = (url_similarity * 0.3) + (anchor_keyword_match * 0.4) + (url_keyword_match * 0.3)
                        
                        all_results.append({
                            'Internal_URL': internal_url,
                            'Target_Keyword': target_keyword,
                            'Opportunity_URL': opp_url,
                            'Anchor_Text': anchor_text,
                            'URL_Similarity': round(url_similarity, 4),
                            'Anchor_Keyword_Match': round(anchor_keyword_match, 4),
                            'URL_Keyword_Match': round(url_keyword_match, 4),
                            'Combined_Score': round(combined_score, 4)
                        })
                        
                        progress_bar.progress((i + 1) / len(opportunity_urls))
                    
                    status_text.text(f"✅ Completed analysis for {internal_url}")
                
                if all_results:
                    all_results.sort(key=lambda x: x['Combined_Score'], reverse=True)
                    results_df = pd.DataFrame(all_results)
                    
                    st.success(f"✅ Analyzed {len(all_results)} opportunity matches")
                    
                    for internal_url in internal_df['Internal_URL'].unique():
                        internal_results = results_df[results_df['Internal_URL'] == internal_url]
                        top_results = internal_results.head(5)
                        
                        st.subheader(f"🏆 Top Opportunities for: {internal_url}")
                        
                        for idx, row in top_results.iterrows():
                            with st.expander(f"#{idx+1} - {row['Opportunity_URL']} (Score: {row['Combined_Score']:.4f})"):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Combined Score", f"{row['Combined_Score']:.4f}")
                                with col2:
                                    st.metric("URL Similarity", f"{row['URL_Similarity']:.4f}")
                                with col3:
                                    st.metric("Anchor-Keyword Match", f"{row['Anchor_Keyword_Match']:.4f}")
                                with col4:
                                    st.metric("URL-Keyword Match", f"{row['URL_Keyword_Match']:.4f}")
                                
                                st.write("**Anchor Text:**")
                                st.write(row['Anchor_Text'])
                    
                    st.subheader("All Results")
                    st.dataframe(results_df)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Opportunity Analysis",
                        data=csv,
                        file_name="opportunity_analysis_results.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()