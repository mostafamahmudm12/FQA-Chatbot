import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import numpy as np
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_data()

# File operations
def create_default_faq_file(filename="faqs.txt"):
    """Create default FAQ file"""
    default_faqs = """# FAQ Data File - Edit this file to customize your FAQs
# Format: Question | Answer

What are your business hours? | Our business hours are Monday to Friday, 9 AM to 6 PM, and Saturday 10 AM to 4 PM. We are closed on Sundays.

How can I track my order? | You can track your order by logging into your account and visiting the 'My Orders' section. You'll receive a tracking number via email once your order ships.

What is your return policy? | We accept returns within 30 days of purchase. Items must be unused and in original packaging. Contact our support team to initiate a return.

Do you offer international shipping? | Yes, we ship to over 50 countries worldwide. Shipping costs and delivery times vary by location.

How do I reset my password? | Click on 'Forgot Password' on the login page. Enter your email address, and we'll send you a link to reset your password.

What payment methods do you accept? | We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and Apple Pay.

How long does shipping take? | Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days.

Can I cancel my order? | Yes, you can cancel your order within 24 hours of placing it. After that, the order may have already been processed.

Do you have a warranty? | All our products come with a 1-year manufacturer warranty covering defects in materials and workmanship.

How do I contact customer support? | You can reach our customer support team via email at support@example.com or call us at 1-800-123-4567.

Are there any discounts available? | Yes! Sign up for our newsletter to receive a 10% discount on your first order.

What is your privacy policy? | We take your privacy seriously. We never share your personal information with third parties.

Do you offer gift cards? | Yes, we offer digital gift cards in denominations of $25, $50, $100, and $200. They never expire!

How do I create an account? | Click 'Sign Up' at the top of our website, enter your email and create a password.

Is my payment information secure? | Absolutely! We use SSL encryption and PCI-compliant payment processing to ensure your information is secure.
"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(default_faqs)
    return True

def load_faqs_from_file(filename="faqs.txt"):
    """Load FAQs from text file"""
    if not os.path.exists(filename):
        create_default_faq_file(filename)
    
    faqs = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '|' in line:
                parts = line.split('|', 1)
                question = parts[0].strip()
                answer = parts[1].strip()
                if question and answer:
                    faqs[question] = answer
    return faqs

def save_faq_to_file(question, answer, filename="faqs.txt"):
    """Add new FAQ to file"""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"\n{question} | {answer}")

def delete_faq_from_file(question_to_delete, filename="faqs.txt"):
    """Delete FAQ from file"""
    faqs = load_faqs_from_file(filename)
    if question_to_delete in faqs:
        del faqs[question_to_delete]
        
        # Rewrite file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# FAQ Data File\n# Format: Question | Answer\n\n")
            for q, a in faqs.items():
                f.write(f"{q} | {a}\n")
        return True
    return False

# Chatbot class
class FAQChatbot:
    def __init__(self, faqs):
        self.faqs = faqs
        self.questions = list(faqs.keys())
        self.answers = list(faqs.values())
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        if self.questions:
            self.preprocessed_questions = [self.preprocess(q) for q in self.questions]
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(self.preprocessed_questions)
    
    def preprocess(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words]
        return ' '.join(tokens)
    
    def get_response(self, user_input, threshold=0.3):
        if not self.questions:
            return "No FAQs available.", 0.0, None
        
        processed_input = self.preprocess(user_input)
        input_vector = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vector, self.tfidf_matrix)[0]
        
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity > threshold:
            return self.answers[best_match_idx], best_similarity, self.questions[best_match_idx]
        else:
            return "I'm sorry, I don't have an answer to that question. Please try rephrasing or contact support.", 0.0, None

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'faqs' not in st.session_state:
    st.session_state.faqs = load_faqs_from_file()

if 'bot' not in st.session_state:
    st.session_state.bot = FAQChatbot(st.session_state.faqs)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .faq-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings & Management")
    
    # Threshold slider
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum similarity score to return an answer"
    )
    
    st.divider()
    
    # FAQ Management
    st.subheader("📋 FAQ Management")
    
    # View all FAQs
    if st.button("📖 View All FAQs", use_container_width=True):
        st.session_state.show_faqs = True
    
    # Add new FAQ
    with st.expander("➕ Add New FAQ"):
        new_question = st.text_input("Question:", key="new_q")
        new_answer = st.text_area("Answer:", key="new_a")
        
        if st.button("Add FAQ", use_container_width=True):
            if new_question and new_answer:
                save_faq_to_file(new_question, new_answer)
                st.session_state.faqs = load_faqs_from_file()
                st.session_state.bot = FAQChatbot(st.session_state.faqs)
                st.success("✅ FAQ added successfully!")
                st.rerun()
            else:
                st.error("Please fill in both question and answer!")
    
    # Delete FAQ
    with st.expander("🗑️ Delete FAQ"):
        if st.session_state.faqs:
            question_to_delete = st.selectbox(
                "Select FAQ to delete:",
                options=list(st.session_state.faqs.keys())
            )
            
            if st.button("Delete Selected FAQ", use_container_width=True):
                if delete_faq_from_file(question_to_delete):
                    st.session_state.faqs = load_faqs_from_file()
                    st.session_state.bot = FAQChatbot(st.session_state.faqs)
                    st.success("✅ FAQ deleted successfully!")
                    st.rerun()
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Statistics")
    st.metric("Total FAQs", len(st.session_state.faqs))
    st.metric("Chat Messages", len(st.session_state.messages))
    
    st.divider()
    
    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Reload FAQs
    if st.button("🔄 Reload FAQs from File", use_container_width=True):
        st.session_state.faqs = load_faqs_from_file()
        st.session_state.bot = FAQChatbot(st.session_state.faqs)
        st.success("✅ FAQs reloaded!")
        st.rerun()

# Main content
st.title("💬 FAQ Chatbot")
st.caption("Powered by NLP and Machine Learning | Data stored in faqs.txt")

# Show all FAQs in main area if requested
if 'show_faqs' in st.session_state and st.session_state.show_faqs:
    st.subheader("📚 All Available FAQs")
    
    for i, (question, answer) in enumerate(st.session_state.faqs.items(), 1):
        with st.container():
            st.markdown(f"""
            <div class="faq-card">
                <strong>Q{i}:</strong> {question}<br>
                <strong>A:</strong> {answer}
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("❌ Close FAQ List"):
        st.session_state.show_faqs = False
        st.rerun()
    
    st.divider()

# Chat interface
st.subheader("💭 Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "confidence" in message and message["confidence"] > 0:
            conf = message["confidence"]
            if conf >= 0.7:
                conf_class = "confidence-high"
            elif conf >= 0.4:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"
            st.markdown(f'<p class="{conf_class}">🎯 Confidence: {conf:.1%}</p>', unsafe_allow_html=True)
        if "matched_question" in message and message["matched_question"]:
            st.caption(f"📌 Matched: {message['matched_question']}")

# Chat input
if prompt := st.chat_input("Ask me anything about our services..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    answer, confidence, matched_question = st.session_state.bot.get_response(prompt, threshold)
    
    # Add assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)
        if confidence > 0:
            conf = confidence
            if conf >= 0.7:
                conf_class = "confidence-high"
            elif conf >= 0.4:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"
            st.markdown(f'<p class="{conf_class}">🎯 Confidence: {conf:.1%}</p>', unsafe_allow_html=True)
        if matched_question:
            st.caption(f"📌 Matched: {matched_question}")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "confidence": confidence,
        "matched_question": matched_question
    })

# Footer
st.divider()
st.caption(f"📁 FAQ File: faqs.txt | ⏰ Last loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")