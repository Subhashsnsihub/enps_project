import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pdfkit
import jinja2
import markdown
import base64
from datetime import datetime
import io

# Set page configuration
st.set_page_config(
    page_title="eNPS Analytics Hub",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container {
        padding: 2rem 3rem;
        background: linear-gradient(135deg, #ffe0b2, #ffab91, #ffcc80); /* Soft peach to light orange to golden yellow */
    }

    h1 {
        background: linear-gradient(45deg, #ffcc80, #ffab91, #ffe0b2); /* Warm golden yellow to soft orange */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        padding: 1rem 0;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    h2 {
        color: #5d4037;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    h3 {
        color: #6d4c41;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        color: #ff7043 !important; /* Vibrant coral */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    div[data-testid="stMetricDelta"] {
        font-size: 1.2rem !important;
        color: #f57c00 !important; /* Amber for growth */
    }

    .stPlotlyChart {
        background: linear-gradient(135deg, #ffe0b2, #ffab91, #ffcc80); /* Soft peach to light orange to golden yellow */
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 1rem;
        margin: 1rem 0;
        transition: transform 0.3s, box-shadow 0.3s;
    }

    .stPlotlyChart:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    .css-1d391kg {
        background: linear-gradient(135deg, #ffe0b2, #ffab91, #ffcc80); /* Soft peach to light orange to golden yellow */
        border-right: 1px solid #ff7043;
        padding: 2rem 1rem;
    }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffe0b2, #ffab91, #ffcc80); /* Soft peach to light orange to golden yellow */
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s, box-shadow 0.3s;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    .report-section {
        background: linear-gradient(135deg, #ffe0b2, #ffab91, #ffcc80); /* Soft peach to light orange to golden yellow */
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .download-button {
        background: linear-gradient(90deg, #ff7043, #f57c00, #ffcc80); /* Vibrant coral to amber to golden yellow */
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        text-decoration: none;
        display: inline-block;
        margin: 1rem 0;
        font-weight: 600;
        transition: background-color 0.3s, transform 0.3s;
    }

    .download-button:hover {
        background: linear-gradient(90deg, #f57c00, #ff7043, #ff5722);
        transform: scale(1.05);
    }

    .custom-divider {
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #ff7043, #ffab91, #ffcc80); /* Gradient of soft peach and coral */
        margin: 2rem 0;
    }

    /* Footer Style */
    .footer {
        background: linear-gradient(135deg, #f57c00, #ff7043); /* Amber to coral */
        padding: 1.5rem 3rem;
        color: #fff;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .footer a {
        color: #ffffff;
        text-decoration: none;
        font-weight: 600;
    }

    .footer a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)





import streamlit as st
import pandas as pd
import io

def load_data():
    """
    Load data either from uploaded file or create sample data if no file is uploaded
    Returns a pandas DataFrame
    """
    # Add file uploader to sidebar
    uploaded_file = st.sidebar.file_uploader(
        "Upload eNPS Data (Excel file)", 
        type=['xlsx', 'xls'],
        help="Upload your analyzed eNPS data file in Excel format"
    )
    
    if uploaded_file is not None:
        try:
            return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()
    else:
        # Create sample data for demonstration
        sample_data = {
            'eNPS Score': [7, 8, 6, 9, 5, 4, 8, 7, 9, 3],
            'eNPS_Category': ['Promoter', 'Promoter', 'Passive', 'Promoter', 'Passive', 
                            'Detractor', 'Promoter', 'Promoter', 'Promoter', 'Detractor'],
            'Feedback_Cleaned': ['Great work environment', 'Good benefits', 'Average experience',
                               'Excellent culture', 'Okay workplace', 'Need improvements',
                               'Love the team', 'Good management', 'Amazing opportunities',
                               'Poor communication'],
            'Dept_Sales': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            'Dept_Engineering': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            'Dept_Marketing': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        st.sidebar.warning("Using sample data. Upload your Excel file for actual analysis.")
        return pd.DataFrame(sample_data)



def create_wordcloud(texts):
    """Generate and return a wordcloud figure"""
    wordcloud = WordCloud(
        background_color='white',
        width=1000,
        height=500,
        max_words=100,
        colormap='viridis',
        prefer_horizontal=0.7
    ).generate(' '.join(texts))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def create_topic_wordclouds(texts):
    """Generate topic-specific wordclouds"""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=1000
    )
    X = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(
        n_components=5,
        random_state=42,
        max_iter=20
    )
    lda.fit(X)
    
    topic_names = [
        "💼 Work Environment",
        "📈 Career Growth",
        "👥 Management",
        "💰 Benefits & Compensation",
        "🌟 Company Culture"
    ]
    
    topic_wordclouds = []
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-15:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        word_freq = {word: topic[idx] for idx, word in zip(top_words_idx, top_words)}
        
        wordcloud = WordCloud(
            background_color='white',
            width=1000,
            height=500,
            colormap='viridis',
            prefer_horizontal=0.7
        ).generate_from_frequencies(word_freq)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{topic_names[topic_idx]}', pad=20, size=16, fontweight='bold')
        topic_wordclouds.append(fig)
    
    return topic_wordclouds

def generate_html_report(df, metrics, dept_analysis, wordcloud_path):
    """Generate an HTML report using a template"""
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>eNPS Analytics Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }
            .header { text-align: center; margin-bottom: 30px; }
            .metric-container { display: flex; justify-content: space-between; margin: 20px 0; }
            .metric-box { 
                background: #f8f9fa; 
                padding: 20px; 
                border-radius: 8px;
                width: 22%;
                text-align: center;
            }
            .section { margin: 40px 0; }
            .chart-container { margin: 20px 0; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            .wordcloud-container { text-align: center; margin: 30px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>eNPS Analytics Report</h1>
            <p>Generated on: {{ generation_date }}</p>
        </div>

        <div class="section">
            <h2>Key Metrics</h2>
            <div class="metric-container">
                {% for metric in metrics %}
                <div class="metric-box">
                    <h3>{{ metric.title }}</h3>
                    <p style="font-size: 24px;">{{ metric.value }}</p>
                    <p style="color: {% if metric.trend > 0 %}green{% else %}red{% endif %}">
                        {{ metric.trend }}%
                    </p>
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <h2>Department Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Department</th>
                        <th>Average Score</th>
                        <th>YoY Change</th>
                    </tr>
                </thead>
                <tbody>
                    {% for dept in dept_analysis %}
                    <tr>
                        <td>{{ dept.name }}</td>
                        <td>{{ dept.score }}</td>
                        <td>{{ dept.change }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Feedback Analysis</h2>
            <div class="wordcloud-container">
                <img src="{{ wordcloud_path }}" alt="Word Cloud" style="max-width: 100%;">
            </div>
        </div>
    </body>
    </html>
    """
    
    env = jinja2.Environment()
    template = env.from_string(template)
    
    return template.render(
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metrics=metrics,
        dept_analysis=dept_analysis,
        wordcloud_path=wordcloud_path
    )

def generate_pdf_report(html_content):
    """Convert HTML report to PDF"""
    options = {
        'page-size': 'A4',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
    }
    try:
        pdf = pdfkit.from_string(html_content, False, options=options)
        return pdf
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def generate_markdown_report(df, metrics, dept_analysis):
    """Generate a Markdown report"""
    markdown_content = f"""
# eNPS Analytics Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Key Metrics
"""
    
    for metric in metrics:
        markdown_content += f"""
### {metric['title']}
* Value: {metric['value']}
* Trend: {metric['trend']}%
"""
    
    markdown_content += "\n## Department Analysis\n"
    markdown_content += "| Department | Average Score | YoY Change |\n"
    markdown_content += "|------------|---------------|------------|\n"
    
    for dept in dept_analysis:
        markdown_content += f"| {dept['name']} | {dept['score']} | {dept['change']}% |\n"
    
    return markdown_content

def download_button(object_to_download, download_filename, button_text):
    """Generate a download button for any object"""
    if isinstance(object_to_download, bytes):
        b64 = base64.b64encode(object_to_download).decode()
    else:
        b64 = base64.b64encode(object_to_download.encode()).decode()
    
    button_uuid = f"download_button_{download_filename}"
    custom_css = f"""
        <style>
            #{button_uuid} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_uuid}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
        </style>
    """
    
    dl_link = (
        custom_css
        + f'<a download="{download_filename}" id="{button_uuid}" href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
    )
    return dl_link

def main():
    # Load data
    df = load_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h1 style='font-size: 1.5rem !important; margin-bottom: 1rem;'>eNPS Analytics Hub</h1>
                <div style='width: 100px; height: 3px; background: linear-gradient(90deg, #3498db, transparent); margin: 0 auto;'></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Dashboard Controls")
        
        # Enhanced filters
        selected_departments = st.multiselect(
            "Select Departments",
            options=[col.replace('Dept_', '') for col in df.columns if col.startswith('Dept_')],
            default=[col.replace('Dept_', '') for col in df.columns if col.startswith('Dept_')]
        )
        
        st.markdown("### 📅 Time Period")
        date_range = st.date_input(
            "Select Date Range",
            value=(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-12-31'))
        )
        
        st.markdown("""
            <div class='status-card' style='margin-top: 2rem;'>
                <h4 style='margin: 0; color: #2c3e50;'>Dashboard Status</h4>
                <p style='margin: 0.5rem 0 0 0; color: #7f8c8d;'>Last updated: Today</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main Content
    st.markdown("""
        <h1>Employee Net Promoter Score Analytics</h1>
        <p style='color: #7f8c8d; font-size: 1.1rem;'>Comprehensive analysis of employee satisfaction and engagement metrics</p>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Main Content (continued)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        promoters = (df['eNPS_Category'] == 'Promoter').mean() * 100
        st.metric("🌟 Promoters", f"{promoters:.1f}%", "+2.5%")
    
    with metric_col2:
        detractors = (df['eNPS_Category'] == 'Detractor').mean() * 100
        st.metric("📉 Detractors", f"{detractors:.1f}%", "-1.2%")
    
    with metric_col3:
        enps = promoters - detractors
        st.metric("📊 eNPS Score", f"{enps:.1f}", "+3.7%")
    
    with metric_col4:
        response_rate = (df['Feedback'].notna().sum() / len(df)) * 100
        st.metric("📝 Response Rate", f"{response_rate:.1f}%", "+5.2%")
    
    
    # Score Distribution and Categories
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Score Distribution Analysis
    with col1:
        st.markdown("### 📈 Score Distribution Analysis")
        fig_dist = px.histogram(
            df,
            x="eNPS Score",
            color="eNPS_Category",
            title="Distribution of eNPS Scores",
            color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c'],
            width=600,
            height=400
        )
        fig_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            title_x=0.5,
            title_font_size=20,
            margin=dict(l=50, r=50, t=80, b=50),
            bargap=0.1,
            xaxis=dict(
                title_text="eNPS Score",
                tickfont=dict(size=12),
                title_font=dict(size=14),
                tickangle=-45  # Rotate the x-axis labels to avoid overlapping
            ),
            yaxis=dict(
                title_text="Count",
                tickfont=dict(size=12),
                title_font=dict(size=14)
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Category Distribution (Pie Chart)
    with col2:
        st.markdown("### 🔄 Category Distribution")
        fig_pie = px.pie(
            df,
            names="eNPS_Category",
            title="Distribution of eNPS Categories",
            color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c'],
            width=600,
            height=400
        )
        fig_pie.update_traces(
            textinfo="label+percent",  # Show label and percentage
            textfont_size=14,         # Adjust font size of labels
            pull=[0.1 if value == df["eNPS_Category"].value_counts().idxmax() else 0 
                for value in df["eNPS_Category"]]
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_x=0.5,
            title_font_size=20,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,  # Move legend below the chart
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    
    # Department Analysis
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Departmental Performance Analysis")
    
    # Calculate department scores more explicitly
    dept_cols = [col for col in df.columns if col.startswith('Dept_')]
    dept_avg_scores = []
    
    for col in dept_cols:
        dept_name = col.replace('Dept_', '')
        dept_data = df[df[col] == 1]
        if len(dept_data) > 0:  
            avg_score = dept_data['eNPS Score'].mean()
            count = len(dept_data)
            dept_avg_scores.append({
                'Department': dept_name,
                'Average Score': round(avg_score, 2),
                'Response Count': count
            })
    
    # Create DataFrame for visualization
    dept_df = pd.DataFrame(dept_avg_scores)
    
    # Create enhanced bar chart
    if not dept_df.empty:
        fig_dept = go.Figure()
        
        # Add main bar chart
        fig_dept.add_trace(go.Bar(
            x=dept_df['Department'],
            y=dept_df['Average Score'],
            text=dept_df['Average Score'].round(1),
            textposition='auto',
            marker_color='#3498db',
            name='Average Score',
            hovertemplate="<b>%{x}</b><br>" +
                         "Average Score: %{y:.1f}<br>" +
                         "Response Count: %{customdata}<extra></extra>",
            customdata=dept_df['Response Count']
        ))
        
        # Update layout with enhanced styling
        fig_dept.update_layout(
            title={
                'text': "Average eNPS Score by Department",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20)
            },
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            width=900,  # Increased width
            height=500,  # Increased height
            margin=dict(l=50, r=50, t=80, b=100),
            xaxis=dict(
                title="Department",
                tickangle=-45,
                tickfont=dict(size=12),
                title_font=dict(size=14)
            ),
            yaxis=dict(
                title="Average eNPS Score",
                tickfont=dict(size=12),
                title_font=dict(size=14),
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='#CCCCCC'
            ),
            showlegend=False,
            bargap=0.2
        )
        
        # Add average line
        overall_avg = dept_df['Average Score'].mean()
        fig_dept.add_hline(
            y=overall_avg,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"Overall Average: {overall_avg:.1f}",
            annotation_position="bottom right"
        )
        
        # Display the chart
        st.plotly_chart(fig_dept, use_container_width=True)
        
        # Add summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Highest Performing Dept",
                dept_df.loc[dept_df['Average Score'].idxmax(), 'Department'],
                f"{dept_df['Average Score'].max():.1f}"
            )
        with col2:
            st.metric(
                "Lowest Performing Dept",
                dept_df.loc[dept_df['Average Score'].idxmin(), 'Department'],
                f"{dept_df['Average Score'].min():.1f}"
            )
        with col3:
            st.metric(
                "Average Department Score",
                f"{overall_avg:.1f}",
                None
            )
    else:
        st.warning("No department data available for visualization.")
    
    # Feedback Analysis Section
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 💭 Employee Feedback Analysis")
    
    overall_wordcloud = create_wordcloud(df['Feedback_Cleaned'].dropna())
    st.pyplot(overall_wordcloud)
    
    # Topic-specific Analysis
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🎯 Topic-Specific Insights")
    
    topic_clouds = create_topic_wordclouds(df['Feedback_Cleaned'].dropna())
    
    for idx in range(0, len(topic_clouds), 2):
        col1, col2 = st.columns(2)
        with col1:
            if idx < len(topic_clouds):
                st.pyplot(topic_clouds[idx])
        with col2:
            if idx + 1 < len(topic_clouds):
                st.pyplot(topic_clouds[idx + 1])
    
    # Report Generation Section
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📑 Generate Reports")
    
    report_col1, report_col2 = st.columns(2)
    
    with report_col1:
        report_format = st.selectbox(
            "Select Report Format",
            ["HTML", "PDF", "Markdown"],
            help="Choose the format for your report"
        )
        
        include_sections = st.multiselect(
            "Select Sections to Include",
            ["Key Metrics", "Department Analysis", "Feedback Analysis", "Statistical Insights"],
            default=["Key Metrics", "Department Analysis", "Feedback Analysis"]
        )
    
    with report_col2:
        report_period = st.date_input(
            "Select Report Period",
            value=(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-12-31')),
            help="Choose the date range for your report"
        )
        
        include_charts = st.checkbox("Include Charts", value=True)
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            # Prepare report data
            metrics = [
                {"title": "Promoters", "value": f"{promoters:.1f}%", "trend": 2.5},
                {"title": "Detractors", "value": f"{detractors:.1f}%", "trend": -1.2},
                {"title": "eNPS Score", "value": f"{enps:.1f}", "trend": 3.7},
                {"title": "Response Rate", "value": f"{response_rate:.1f}%", "trend": 5.2}
            ]
            
            dept_analysis = [
                {"name": dept["Department"], "score": f"{dept['Average Score']:.1f}", "change": 15}
                for dept in dept_avg_scores
            ]
            
            # Generate report based on selected format
            if report_format == "HTML":
                html_report = generate_html_report(df, metrics, dept_analysis, "wordcloud.png")
                st.markdown(
                    download_button(html_report, "enps_report.html", "📥 Download HTML Report"),
                    unsafe_allow_html=True
                )
            
            elif report_format == "PDF":
                html_report = generate_html_report(df, metrics, dept_analysis, "wordcloud.png")
                pdf_report = generate_pdf_report(html_report)
                if pdf_report:
                    st.markdown(
                        download_button(pdf_report, "enps_report.pdf", "📥 Download PDF Report"),
                        unsafe_allow_html=True
                    )
            
            else:  # Markdown
                markdown_report = generate_markdown_report(df, metrics, dept_analysis)
                st.markdown(
                    download_button(markdown_report, "enps_report.md", "📥 Download Markdown Report"),
                    unsafe_allow_html=True
                )
            
            st.success("Report generated successfully!")
   
   # Footer
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #004d40, #00897b); border-radius: 10px;'>
            <p style='color: #ffffff; font-size: 1rem; font-weight: 600;'>eNPS Analysis Dashboard | Updated: January 2024</p>
            <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; color: #ffffff;'>
                <span>📊 Data Refresh: Daily</span>
                <span>👥 Responses: {len(df):,}</span>
                <span>📈 Trend: Positive</span>
            </div>
            <div style='margin-top: 2rem; color: #ffffff; font-size: 0.9rem;'>
                <p>Thank you for visiting!</p>
                <p>For more info, visit <a href="https://yourwebsite.com" style="color: #ffffff; text-decoration: underline;">Our Website</a></p>
            </div>
        </div>
    """, unsafe_allow_html=True)




if __name__ == "__main__":
    main()
