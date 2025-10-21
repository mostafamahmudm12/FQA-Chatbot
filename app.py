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
from openai import OpenAI
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Smart FAQ Chatbot",
    page_icon="🤖",
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

# ================= FILE OPERATIONS =================

def create_default_faq_file(filename="faqs.txt"):
    """Create default FAQ file"""
    default_faqs = """# FAQ Data File - Your company-specific FAQs
# Format: Question | Answer
# The bot will use these first, then use OpenAI for other questions

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

def get_faq_context(faqs):
    """Convert FAQs to context string for OpenAI"""
    context = "Company FAQs:\n\n"
    for q, a in faqs.items():
        context += f"Q: {q}\nA: {a}\n\n"
    return context

# ================= OPENAI CHATBOT CLASS =================

class OpenAIChatbot:
    def __init__(self, faqs, api_key, use_faq_matching=True, threshold=0.3):
        self.faqs = faqs
        self.questions = list(faqs.keys())
        self.answers = list(faqs.values())
        self.api_key = api_key
        self.use_faq_matching = use_faq_matching
        self.threshold = threshold
        self.client = None
        
        # Initialize OpenAI client
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:
                st.error(f"Error initializing OpenAI: {str(e)}")
        
        # Initialize FAQ matching
        if use_faq_matching and self.questions:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self.preprocessed_questions = [self.preprocess(q) for q in self.questions]
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(self.preprocessed_questions)
    
    def preprocess(self, text):
        """Preprocess text for FAQ matching"""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words]
        return ' '.join(tokens)
    
    def get_faq_match(self, user_input):
        """Check if question matches FAQ database"""
        if not self.use_faq_matching or not self.questions:
            return None, 0.0, None
        
        processed_input = self.preprocess(user_input)
        input_vector = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vector, self.tfidf_matrix)[0]
        
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity > self.threshold:
            return self.answers[best_match_idx], best_similarity, self.questions[best_match_idx]
        return None, best_similarity, None
    
    def get_openai_response(self, user_input, conversation_history=None):
        """Get response from OpenAI GPT"""
        if not self.client:
            return "Please enter a valid OpenAI API key in the sidebar."
        
        try:
            # Build messages with FAQ context
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful customer service assistant. 
                    
Here are our company FAQs for reference:
{get_faq_context(self.faqs)}

When answering questions:
1. If the question is about our company/products/services, use the FAQ information above
2. For general questions not in FAQs, provide helpful, accurate answers
3. Be friendly, professional, and concise
4. If you're not sure about company-specific details, suggest contacting support"""
                }
            ]
            
            # Add conversation history if available
            if conversation_history:
                for msg in conversation_history[-6:]:  # Last 3 exchanges
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current question
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error: {str(e)}\n\nPlease check your API key and internet connection."
    
    def get_response(self, user_input, conversation_history=None):
        """
        Main response method - tries FAQ first, then OpenAI
        """
        # Try FAQ match first if enabled
        if self.use_faq_matching:
            faq_answer, confidence, matched_question = self.get_faq_match(user_input)
            
            if faq_answer:
                return {
                    "answer": faq_answer,
                    "confidence": confidence,
                    "matched_question": matched_question,
                    "source": "FAQ Database",
                    "tokens_used": 0
                }
        
        # Use OpenAI for everything else
        openai_answer = self.get_openai_response(user_input, conversation_history)
        
        return {
            "answer": openai_answer,
            "confidence": 0.85,  # High confidence for OpenAI
            "matched_question": None,
            "source": "OpenAI GPT",
            "tokens_used": len(user_input.split()) + len(openai_answer.split())
        }

# ================= STREAMLIT APP =================

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    .source-faq {
        background-color: #d4edda;
        color: #155724;
    }
    .source-openai {
        background-color: #d1ecf1;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'faqs' not in st.session_state:
    st.session_state.faqs = load_faqs_from_file()

if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0

# Sidebar
with st.sidebar:
    st.title("⚙️ OpenAI Configuration")
    
    # API Key Input
    st.subheader("🔑 API Key")
    api_key = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="Get your API key from: https://platform.openai.com/api-keys"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key to start chatting")
        st.markdown("""
        **How to get API key:**
        1. Go to [OpenAI Platform](https://platform.openai.com/signup)
        2. Sign up / Log in
        3. Go to [API Keys](https://platform.openai.com/api-keys)
        4. Click "Create new secret key"
        5. Copy and paste it above
        """)
    else:
        st.success("✅ API key configured")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    
    use_faq_matching = st.checkbox(
        "Enable FAQ Matching",
        value=True,
        help="Check FAQs first before using OpenAI (saves tokens)"
    )
    
    threshold = st.slider(
        "FAQ Match Threshold",
        0.0, 1.0, 0.3, 0.05,
        help="Higher = Stricter FAQ matching, Lower = Use OpenAI more often"
    )
    
    model_temp = st.slider(
        "Response Creativity",
        0.0, 1.0, 0.7, 0.1,
        help="Higher = More creative, Lower = More focused"
    )
    
    st.divider()
    
    # FAQ Management
    st.subheader("📋 FAQ Management")
    
    with st.expander("➕ Add New FAQ"):
        new_q = st.text_input("Question:", key="new_q")
        new_a = st.text_area("Answer:", key="new_a")
        
        if st.button("Add FAQ", use_container_width=True):
            if new_q and new_a:
                save_faq_to_file(new_q, new_a)
                st.session_state.faqs = load_faqs_from_file()
                st.success("✅ FAQ added!")
                st.rerun()
            else:
                st.error("Please fill both fields")
    
    with st.expander("📖 View All FAQs"):
        if st.session_state.faqs:
            for i, (q, a) in enumerate(st.session_state.faqs.items(), 1):
                st.markdown(f"**{i}. {q}**")
                st.caption(a)
                st.divider()
        else:
            st.info("No FAQs available")
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Statistics")
    st.metric("Total FAQs", len(st.session_state.faqs))
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Est. Tokens Used", st.session_state.total_tokens)
    
    estimated_cost = (st.session_state.total_tokens / 1000) * 0.002
    st.metric("Est. Cost", f"${estimated_cost:.4f}")
    
    st.divider()
    
    # Actions
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.rerun()
    
    if st.button("🔄 Reload FAQs", use_container_width=True):
        st.session_state.faqs = load_faqs_from_file()
        st.success("✅ Reloaded!")
        st.rerun()

# Main content
st.title("🤖 Smart FAQ Chatbot")
st.caption("💡 Powered by OpenAI GPT-3.5-turbo + FAQ Database")

# Info box
if api_key:
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **Intelligent Hybrid System:**
        
        1. 🔍 **FAQ Check**: First searches your FAQ database for matches
        2. 🤖 **OpenAI Backup**: If no FAQ match, uses OpenAI to answer ANY question
        3. 💰 **Cost Efficient**: FAQ answers are FREE, only pay for OpenAI usage
        
        **Benefits:**
        - Instant answers for common questions (FAQ)
        - Can answer ANY question (OpenAI)
        - Maintains conversation context
        - Saves money by using FAQs first
        """)

# Initialize bot
if 'bot' not in st.session_state or st.session_state.get('current_api_key') != api_key:
    if api_key:
        st.session_state.bot = OpenAIChatbot(
            st.session_state.faqs, 
            api_key,
            use_faq_matching,
            threshold
        )
        st.session_state.current_api_key = api_key

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if "metadata" in msg and msg["role"] == "assistant":
            meta = msg["metadata"]
            
            # Source badge
            if meta["source"] == "FAQ Database":
                st.markdown(
                    '<span class="source-badge source-faq">✅ FAQ Database (Free)</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span class="source-badge source-openai">🤖 OpenAI GPT</span>',
                    unsafe_allow_html=True
                )
            
            # Show matched question if from FAQ
            if meta.get("matched_question"):
                st.caption(f"📌 Matched: {meta['matched_question']}")
            
            # Show confidence
            if meta.get("confidence") and meta["confidence"] > 0:
                st.caption(f"🎯 Confidence: {meta['confidence']:.1%}")

# Chat input
if prompt := st.chat_input("Ask me anything..." if api_key else "Please enter API key first..."):
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.bot.get_response(
                    prompt,
                    st.session_state.messages
                )
            
            st.markdown(result["answer"])
            
            # Source badge
            if result["source"] == "FAQ Database":
                st.markdown(
                    '<span class="source-badge source-faq">✅ FAQ Database (Free)</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span class="source-badge source-openai">🤖 OpenAI GPT</span>',
                    unsafe_allow_html=True
                )
            
            # Show matched question
            if result.get("matched_question"):
                st.caption(f"📌 Matched: {result['matched_question']}")
            
            # Show confidence
            if result.get("confidence"):
                st.caption(f"🎯 Confidence: {result['confidence']:.1%}")
        
        # Update token count
        st.session_state.total_tokens += result.get("tokens_used", 0)
        
        # Save message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "metadata": result
        })

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📁 FAQs: {len(st.session_state.faqs)}")
with col2:
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")
with col3:
    st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")