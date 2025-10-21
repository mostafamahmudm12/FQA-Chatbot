import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import numpy as np
import json 
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
def create_default_faq_file(filename="faqs.txt"):
    """Create a default FAQ file if it doesn't exist"""
    default_faqs = """# FAQ Data File
# Format: Question | Answer
# Lines starting with # are comments

What are your business hours? | Our business hours are Monday to Friday, 9 AM to 6 PM, and Saturday 10 AM to 4 PM. We are closed on Sundays.

How can I track my order? | You can track your order by logging into your account and visiting the 'My Orders' section. You'll receive a tracking number via email once your order ships.

What is your return policy? | We accept returns within 30 days of purchase. Items must be unused and in original packaging. Contact our support team to initiate a return.

Do you offer international shipping? | Yes, we ship to over 50 countries worldwide. Shipping costs and delivery times vary by location. Check our shipping page for details.

How do I reset my password? | Click on 'Forgot Password' on the login page. Enter your email address, and we'll send you a link to reset your password.

What payment methods do you accept? | We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and Apple Pay.

How long does shipping take? | Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. International shipping may take 10-15 business days.

Can I cancel my order? | Yes, you can cancel your order within 24 hours of placing it. After that, the order may have already been processed and shipped.

Do you have a warranty? | All our products come with a 1-year manufacturer warranty covering defects in materials and workmanship.

How do I contact customer support? | You can reach our customer support team via email at support@example.com or call us at 1-800-123-4567 during business hours.

Are there any discounts available? | Yes! Sign up for our newsletter to receive a 10% discount on your first order. We also run seasonal promotions.

What is your privacy policy? | We take your privacy seriously. We never share your personal information with third parties. Visit our Privacy Policy page for full details.

Do you offer gift cards? | Yes, we offer digital gift cards in denominations of $25, $50, $100, and $200. They never expire!

How do I create an account? | Click 'Sign Up' at the top of our website, enter your email and create a password. You'll receive a confirmation email to verify your account.

who did it to you? | its Mostafa Mahmoud. 

Is my payment information secure? | Absolutely! We use SSL encryption and PCI-compliant payment processing to ensure your payment information is always secure.
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(default_faqs)
    print(f"✓ Created default FAQ file: {filename}")


def load_faqs_from_file(filename="faqs.txt"):
    """Load FAQs from text file"""
    faqs = {}
    
    # Create file if it doesn't exist
    if not os.path.exists(filename):
        print(f"FAQ file not found. Creating {filename}...")
        create_default_faq_file(filename)
    
    # Read FAQs from file
    with open(filename, 'r', encoding='utf-8') as f:
        line_number = 0
        for line in f:
            line_number += 1
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check if line has proper format
            if '|' not in line:
                print(f"⚠ Warning: Line {line_number} is not properly formatted. Skipping...")
                continue
            
            # Split question and answer
            parts = line.split('|', 1)  # Split only on first |
            question = parts[0].strip()
            answer = parts[1].strip()
            
            if question and answer:
                faqs[question] = answer
            else:
                print(f"⚠ Warning: Line {line_number} has empty question or answer. Skipping...")
    
    print(f"✓ Loaded {len(faqs)} FAQs from {filename}")
    return faqs


def save_faq_to_file(question, answer, filename="faqs.txt"):
    """Add a new FAQ to the file"""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"\n{question} | {answer}")
    print(f"✓ New FAQ added to {filename}")


class FQAFileManager:
    """Handles loading and saving FAQs to a text file"""
    @staticmethod
    def faqs_to_file(faqs,filename='faqs.txt'):

        with open(filename,'w',encoding='utf-8') as f:
            for question,answer  in faqs.items():
                f.write(f"{question} | {answer}\n")

        print(f"✓ FAQs saved to {filename}")


    @staticmethod
    def load_faqs_from_file(filename="faqs.txt"):
        faqs={}
        if not os.path.exists(filename):
            print(f"✗ File {filename} not found!")
            return faqs
        
        with open(filename,'r',encoding='utf-8') as f:
            for line_num, line in enumerate(f,1):
                line=line.strip()
                if not line or line.startswith('#'):
                    continue

                # Split by | separator
                if '|' in line:
                    parts=line.split('|',1)
                    question =parts[0].strip()
                    answer =parts[1].strip()
                    faqs[question]=answer
                else:
                    print(f"⚠ Warning: Line {line_num} doesn't have proper format (Question | Answer)")
                
        print(f"✓ Loaded {len(faqs)} FAQs from {filename}")
        return faqs
    
    def save_faqs_to_json(faqs,filename="faqs.json"):
        """Save FAQs to JSON format (alternative format)"""
        with open(filename,'w',encoding='utf-8') as f:
            json.dump(faqs,f,indent=4,ensure_ascii=False)
        print(f"✓ FAQs saved to {filename}")


    @staticmethod
    def load_faqs_from_json(filename="faqs.json"):
        """Load FAQs from JSON format"""
        if not os.path.exists(filename):
            print(f"✗ File {filename} not found!")
            return {}
        
        with open(filename,'r',encoding='utf-8') as f:
            faqs= json.load(f)

        print(f"✓ Loaded {len(faqs)} FAQs from {filename}")
        return faqs
    

class FAQChatbot:
    def __init__(self, faqs):
        """Initialize the chatbot with FAQ data from file"""
        if not faqs:
            print("✗ Error: No FAQs loaded! Please check your faqs.txt file.")
            return
        
        self.faqs = faqs
        self.questions = list(faqs.keys())
        self.answers = list(faqs.values())
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Preprocess all FAQ questions
        self.preprocessed_questions = [self.preprocess(q) for q in self.questions]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.preprocessed_questions)
    
    def preprocess(self, text):
        """Preprocess text: lowercase, tokenize, remove stopwords, lemmatize"""
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def get_response(self, user_input, threshold=0.3):
        """Find the best matching FAQ and return the answer"""
        # Preprocess user input
        processed_input = self.preprocess(user_input)
        
        # Convert to TF-IDF vector
        input_vector = self.vectorizer.transform([processed_input])
        
        # Calculate cosine similarity with all FAQ questions
        similarities = cosine_similarity(input_vector, self.tfidf_matrix)[0]
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Return answer if similarity is above threshold
        if best_similarity > threshold:
            return self.answers[best_match_idx], best_similarity, self.questions[best_match_idx]
        else:
            return "I'm sorry, I don't have an answer to that question. Please try rephrasing or contact support.", 0.0, None
    
    def chat(self):
        """Simple command-line chat interface"""
        print("=" * 70)
        print("FAQ CHATBOT - Powered by faqs.txt file")
        print("=" * 70)
        print("Commands:")
        print("  - Type your question to get an answer")
        print("  - Type 'list' to see all available FAQs")
        print("  - Type 'add' to add a new FAQ")
        print("  - Type 'quit' or 'exit' to end")
        print("=" * 70 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            # Exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Goodbye! Have a great day!")
                break
            
            # List all FAQs
            elif user_input.lower() == 'list':
                print("\n📋 Available FAQs:")
                print("-" * 70)
                for i, question in enumerate(self.questions, 1):
                    print(f"{i}. {question}")
                print("-" * 70 + "\n")
                continue
            
            # Add new FAQ
            elif user_input.lower() == 'add':
                print("\n➕ Add New FAQ")
                new_question = input("Enter question: ").strip()
                new_answer = input("Enter answer: ").strip()
                
                if new_question and new_answer:
                    save_faq_to_file(new_question, new_answer)
                    self.faqs[new_question] = new_answer
                    self.questions.append(new_question)
                    self.answers.append(new_answer)
                    
                    # Rebuild the model
                    self.preprocessed_questions.append(self.preprocess(new_question))
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.preprocessed_questions)
                    print("✓ FAQ added successfully!\n")
                else:
                    print("✗ Question and answer cannot be empty!\n")
                continue
            
            # Empty input
            if not user_input:
                print("Bot: Please ask a question.\n")
                continue
            
            # Get response
            answer, confidence, matched_question = self.get_response(user_input)
            print(f"\nBot: {answer}")
            if matched_question:
                print(f"📌 Matched Question: {matched_question}")
            print(f"🎯 Confidence: {confidence:.2%}\n")


if __name__ == "__main__":
    print("\n🤖 Starting FAQ Chatbot...")
    print("=" * 70)
    
    # Load FAQs from file
    faqs = load_faqs_from_file("faqs.txt")
    
    if faqs:
        # Create and run chatbot
        bot = FAQChatbot(faqs)
        bot.chat()
    else:
        print("✗ No FAQs available. Please add FAQs to faqs.txt file.")
        print("Format: Question | Answer (one per line)")